import argparse
import os
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch

from NeuGN.graph_tokenizer import GraphTokenizer
from NeuGN.model import GraphDecoder
from NeuGN.nx_utils import graph2path_v2
from NeuGN.pt_graph import PTBatch, PTGraph
from NeuGN.utils import load_model_args, load_ori_graph, load_value2id, save_value2id

SUPPORTED_DATASETS = {"hamster", "lastfm", "wikics", "nell", "dblp", "youtube"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export NeuGN weights/fixtures for pure CUDA inference.")
    p.add_argument("--config_path", default="./model_params/wikics", type=str)
    p.add_argument("--checkpoint", default=None, type=str)
    p.add_argument("--graph_path", default=None, type=str)
    p.add_argument("--dataset", default=None, choices=sorted(SUPPORTED_DATASETS), type=str)
    p.add_argument("--out_dir", default="./cuda_export/wikics", type=str)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--device", default=None, type=str)
    p.add_argument("--batch_size", default=1, type=int)
    p.add_argument("--query_size", default=20, type=int)
    p.add_argument("--seq_len", default=None, type=int)
    p.add_argument("--dump_text", action="store_true")
    return p.parse_args()


def infer_dataset_name(dataset: Optional[str], graph_path: str) -> str:
    if dataset is not None:
        return dataset
    lower = graph_path.lower()
    for name in SUPPORTED_DATASETS:
        if name in lower:
            return name
    raise ValueError(f"Cannot infer dataset from graph_path={graph_path}; please pass --dataset")


def build_adjacency(num_nodes: int, edge_index: torch.Tensor) -> List[Set[int]]:
    adj = [set() for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for s, d in zip(src, dst):
        if s != d:
            adj[s].add(d)
    return adj


def sample_connected_induced_query(
    data_adj: List[Set[int]], data_labels: torch.Tensor, query_size: int, rng: random.Random
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    n = len(data_adj)
    if query_size > n:
        raise ValueError(f"query_size={query_size} > data_nodes={n}")

    for _ in range(300):
        start = rng.randrange(n)
        selected = {start}
        frontier = [start]

        while len(selected) < query_size and frontier:
            u = frontier.pop(0)
            nbrs = list(data_adj[u])
            rng.shuffle(nbrs)
            for v in nbrs:
                if v not in selected:
                    selected.add(v)
                    frontier.append(v)
                    if len(selected) >= query_size:
                        break

            if not frontier and len(selected) < query_size:
                pool = list(selected)
                rng.shuffle(pool)
                for p in pool:
                    for v in data_adj[p]:
                        if v not in selected:
                            selected.add(v)
                            frontier.append(v)
                            break
                    if len(selected) >= query_size:
                        break

        if len(selected) < query_size:
            continue

        orig_nodes = sorted(selected)
        local_map = {nid: i for i, nid in enumerate(orig_nodes)}
        edges = []
        for u in orig_nodes:
            for v in data_adj[u]:
                if v in selected:
                    edges.append((local_map[u], local_map[v]))

        if query_size > 1 and not edges:
            continue

        q_adj = [set() for _ in range(query_size)]
        for u, v in edges:
            q_adj[u].add(v)

        seen = {0}
        stack = [0]
        while stack:
            x = stack.pop()
            for y in q_adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        if len(seen) != query_size:
            continue

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        labels = data_labels[torch.tensor(orig_nodes, dtype=torch.long)].clone()
        return edge_index, labels, orig_nodes

    raise RuntimeError("Failed to sample connected query fixture")


def sanitize_name(name: str) -> str:
    return name.replace(".", "__")


def write_tensor_bin(path: str, t: torch.Tensor) -> None:
    t = t.contiguous().cpu()
    if t.dtype == torch.float32:
        data = t.numpy().astype("float32")
    elif t.dtype == torch.int64:
        data = t.numpy().astype("int64")
    else:
        data = t.float().numpy().astype("float32")
    data.tofile(path)


def export_weights(model: GraphDecoder, out_dir: str) -> None:
    weights_dir = os.path.join(out_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.tsv")

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for name, tensor in model.state_dict().items():
            t = tensor.detach().float().cpu().contiguous()
            rel = f"weights/{sanitize_name(name)}.bin"
            abs_path = os.path.join(out_dir, rel)
            t.numpy().tofile(abs_path)
            shape_csv = ",".join(str(x) for x in t.shape)
            mf.write(f"{name}\tfloat32\t{shape_csv}\t{rel}\n")


def write_shape(path: str, shape: Sequence[int]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(str(int(x)) for x in shape))


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Only batch_size=1 is currently supported")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "input"), exist_ok=True)

    params = load_model_args(args.config_path)
    if params.encoder_config.encoder_name != "gcn":
        raise RuntimeError(f"Only encoder_name=gcn is supported, got {params.encoder_config.encoder_name}")
    if params.decoder_config.decoder_type != "llama":
        raise RuntimeError(f"Only decoder_type=llama is supported, got {params.decoder_config.decoder_type}")

    graph_path = args.graph_path if args.graph_path else params.graph_path
    dataset = infer_dataset_name(args.dataset, graph_path)

    node_values, node_values_uni, edge_src_ids, edge_dst_ids = load_ori_graph(graph_path, dataset)

    mapping_csv = os.path.join(args.config_path, f"{dataset}_value2id_mapping.csv")
    if os.path.exists(mapping_csv):
        value2id = load_value2id(args.config_path, dataset)
    else:
        value2id = save_value2id(node_values_uni, args.config_path, dataset)

    edge_index = torch.stack(
        [torch.tensor(edge_src_ids, dtype=torch.long), torch.tensor(edge_dst_ids, dtype=torch.long)],
        dim=0,
    )
    graph = PTGraph(edge_index=edge_index, num_nodes=len(node_values))
    graph.feat_id = torch.tensor([value2id[str(v)] for v in node_values], dtype=torch.long)

    tokenizer = GraphTokenizer(graph)
    params.decoder_config.vocab_size = tokenizer.token_nums()
    params.encoder_config.graph_value_num = len(node_values_uni)
    params.device = device

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = os.path.join(params.checkpoint_path, f"{params.encoder_config.encoder_name}_checkpoint")

    model = GraphDecoder(params).to(device)
    raw = torch.load(checkpoint, map_location=device)
    state = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    normalized = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    load_res = model.load_state_dict(normalized, strict=False)
    print(f"[load] missing_keys={load_res.missing_keys}")
    print(f"[load] unexpected_keys={load_res.unexpected_keys}")

    keys = set(normalized.keys())
    if not any(k.startswith("encoder.") for k in keys):
        raise RuntimeError("Missing encoder.* weights in checkpoint")
    if not any(k.startswith("decoder.") for k in keys):
        raise RuntimeError("Missing decoder.* weights in checkpoint")

    model.eval()

    data_adj = build_adjacency(graph.num_nodes, graph.edge_index)
    q_edge, q_labels, q_orig_nodes = sample_connected_induced_query(data_adj, graph.feat_id, args.query_size, random.Random(args.seed))

    q_pt = PTGraph(edge_index=q_edge, num_nodes=q_labels.numel())
    q_pt.feat_id = q_labels
    batch_q = PTBatch.from_data_list([q_pt]).to(device)

    if q_pt.num_nodes == 1:
        path_nodes = [0]
    else:
        path_edges = graph2path_v2(q_pt)
        path_nodes = [0] if len(path_edges) == 0 else [u for u, _ in path_edges] + [path_edges[-1][1]]

    if args.seq_len is not None:
        path_nodes = path_nodes[: args.seq_len]

    target = path_nodes[0]
    mapped_set = set(path_nodes[1 : 1 + max(1, len(path_nodes) // 2)])
    mapped_query_to_data = {}
    for qn in mapped_set:
        if qn < len(q_orig_nodes):
            mapped_query_to_data[qn] = q_orig_nodes[qn]

    subnode_id_size = params.decoder_config.sub_node_id_size
    nodeid2sub = {qn: (idx % subnode_id_size) for idx, qn in enumerate(path_nodes)}

    masked_tokens = []
    path_subnode = []
    for qn in path_nodes:
        if qn == target:
            token = tokenizer.sos_id
        elif qn in mapped_query_to_data:
            token = mapped_query_to_data[qn]
        else:
            token = tokenizer.padding_id
        masked_tokens.append(token)
        path_subnode.append(nodeid2sub[qn])

    input_seq = [tokenizer.sos_id] + masked_tokens
    subnode_seq = [nodeid2sub[target]] + path_subnode
    token_mask_len = [len(input_seq)]

    tokens = torch.tensor([input_seq], dtype=torch.long, device=device)
    subnode_ids = torch.tensor([subnode_seq], dtype=torch.long, device=device)
    token_mask_len_t = torch.tensor(token_mask_len, dtype=torch.long, device=device)

    input_dir = os.path.join(args.out_dir, "input")
    write_tensor_bin(os.path.join(input_dir, "graph_edge_index.bin"), batch_q.edge_index.cpu().long())
    write_tensor_bin(os.path.join(input_dir, "graph_feat_id.bin"), batch_q.feat_id.cpu().long())
    write_tensor_bin(os.path.join(input_dir, "graph_batch.bin"), batch_q.batch.cpu().long())
    write_tensor_bin(os.path.join(input_dir, "graph_ptr.bin"), batch_q.ptr.cpu().long())
    write_tensor_bin(os.path.join(input_dir, "tokens.bin"), tokens.cpu().long())
    write_tensor_bin(os.path.join(input_dir, "subnode_ids.bin"), subnode_ids.cpu().long())
    write_tensor_bin(os.path.join(input_dir, "token_mask_len.bin"), token_mask_len_t.cpu().long())

    with open(os.path.join(input_dir, "metadata.txt"), "w", encoding="utf-8") as f:
        f.write(f"dataset={dataset}\n")
        f.write(f"query_size={q_pt.num_nodes}\n")
        f.write(f"query_edges={q_edge.size(1)}\n")
        f.write(f"path_len={len(path_nodes)}\n")
        f.write(f"target_query_node={target}\n")

    export_weights(model, args.out_dir)

    n_kv = params.decoder_config.n_kv_heads if params.decoder_config.n_kv_heads is not None else params.decoder_config.n_heads
    cfg_lines = {
        "encoder_name": params.encoder_config.encoder_name,
        "decoder_type": params.decoder_config.decoder_type,
        "graph_feature_dim": params.encoder_config.graph_feature_dim,
        "graph_value_num": params.encoder_config.graph_value_num,
        "encoder_layers": params.encoder_config.encoder_layers,
        "encoder_hidden_size": params.encoder_config.encoder_hidden_size,
        "decoder_dim": params.decoder_config.dim,
        "n_layers": params.decoder_config.n_layers,
        "n_heads": params.decoder_config.n_heads,
        "n_kv_heads": n_kv,
        "vocab_size": params.decoder_config.vocab_size,
        "multiple_of": params.decoder_config.multiple_of,
        "norm_eps": params.decoder_config.norm_eps,
        "rms_norm_eps": params.decoder_config.rms_norm_eps,
        "pos_size": params.decoder_config.pos_size,
        "sub_node_id_size": params.decoder_config.sub_node_id_size,
        "num_nodes": q_pt.num_nodes,
        "num_edges": int(q_edge.size(1)),
        "batch_size": 1,
        "token_len": len(input_seq),
        "token_mask_len": int(token_mask_len[0]),
        "output_dim": params.decoder_config.vocab_size,
    }
    with open(os.path.join(args.out_dir, "config.txt"), "w", encoding="utf-8") as f:
        for k, v in cfg_lines.items():
            f.write(f"{k}={v}\n")

    with torch.no_grad():
        graph_features = model.get_encoder_tensor(batch_q, device)
        full_logits = model.get_decoder_output(graph_features, tokens, subnode_ids, token_mask_len_t)
        # CUDA path currently validates decoder output head fed by graph feature token.
        cuda_target_logits = model.decoder.output(graph_features[:, 0, :]).unsqueeze(1)

    full_logits = full_logits.detach().float().contiguous().cpu()
    logits = cuda_target_logits.detach().float().contiguous().cpu()
    graph_features = graph_features.detach().float().contiguous().cpu()

    write_tensor_bin(os.path.join(args.out_dir, "python_output.bin"), logits)
    write_shape(os.path.join(args.out_dir, "python_output.shape"), list(logits.shape))
    write_tensor_bin(os.path.join(args.out_dir, "python_full_output.bin"), full_logits)
    write_shape(os.path.join(args.out_dir, "python_full_output.shape"), list(full_logits.shape))
    write_tensor_bin(os.path.join(args.out_dir, "python_graph_features.bin"), graph_features)
    write_shape(os.path.join(args.out_dir, "python_graph_features.shape"), list(graph_features.shape))

    if args.dump_text:
        with open(os.path.join(args.out_dir, "python_output_head.txt"), "w", encoding="utf-8") as f:
            flat = logits.reshape(-1)
            for i in range(min(32, flat.numel())):
                f.write(f"{i}\t{float(flat[i])}\n")

    flat_logits = logits.reshape(-1)
    print(f"checkpoint={checkpoint}")
    print(f"out_dir={args.out_dir}")
    print(f"vocab_size={params.decoder_config.vocab_size}")
    print(f"token_len={len(input_seq)}")
    print(f"python_output_shape={tuple(logits.shape)}")
    print(f"first_10_logits={[float(x) for x in flat_logits[:10]]}")


if __name__ == "__main__":
    main()
