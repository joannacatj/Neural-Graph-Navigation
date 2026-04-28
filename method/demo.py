import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from statistics import mean, median
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch

from NeuGN.graph_tokenizer import GraphTokenizer
from NeuGN.model import GraphDecoder
from NeuGN.nx_utils import graph2path_v2
from NeuGN.pt_graph import PTBatch, PTGraph
from NeuGN.utils import (
    load_model_args,
    load_ori_graph,
    save_value2id,
    load_value2id,
)


SUPPORTED_DATASETS = {"hamster", "lastfm", "wikics", "nell", "dblp", "youtube"}


@dataclass
class QuerySample:
    query_id: int
    edge_index: torch.Tensor  # [2, E]
    labels: torch.Tensor      # [N]
    orig_nodes: List[int]


@dataclass
class MatchResult:
    found: bool
    fms: int
    first_match: Optional[Dict[int, int]]
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NeuGN demo: Python exact subgraph matching evaluator")
    parser.add_argument("--config_path", default="./model_params/wikics", type=str)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--graph_path", default=None, type=str)
    parser.add_argument("--dataset", default=None, type=str, choices=sorted(SUPPORTED_DATASETS))

    parser.add_argument("--query_size", default=16, type=int)
    parser.add_argument("--num_queries", default=20, type=int)
    parser.add_argument("--nav_depth", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--output", default="./demo_results.csv", type=str)

    parser.add_argument("--save_queries", default=None, type=str)
    parser.add_argument("--load_queries", default=None, type=str)

    parser.add_argument("--time_budget", default=None, type=float)
    parser.add_argument("--check_completeness", action="store_true")
    parser.add_argument("--max_steps", default=None, type=int)
    parser.add_argument("--max_matches", default=None, type=int)
    return parser.parse_args()


def infer_dataset_name(dataset: Optional[str], graph_path: str) -> str:
    if dataset is not None:
        return dataset
    gpath = graph_path.lower()
    for name in SUPPORTED_DATASETS:
        if name in gpath:
            return name
    raise ValueError(f"Cannot infer dataset from graph_path={graph_path}. Please pass --dataset explicitly.")


def build_adjacency(num_nodes: int, edge_index: torch.Tensor) -> List[Set[int]]:
    adj: List[Set[int]] = [set() for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for s, d in zip(src, dst):
        if s == d:
            continue
        adj[s].add(d)
    return adj


def load_data_graph(args: argparse.Namespace, device: torch.device):
    params = load_model_args(args.config_path)

    graph_path = args.graph_path if args.graph_path else params.graph_path
    dataset_name = infer_dataset_name(args.dataset, graph_path)

    node_values, node_values_uni, edge_src_ids, edge_dst_ids = load_ori_graph(graph_path, dataset_name)

    mapping_file = os.path.join(args.config_path, f"{dataset_name}_value2id_mapping.csv")
    if os.path.exists(mapping_file):
        value2id = load_value2id(args.config_path, dataset_name)
    else:
        value2id = save_value2id(node_values_uni, args.config_path, dataset_name)

    edge_index_src = torch.tensor(edge_src_ids, dtype=torch.long)
    edge_index_dst = torch.tensor(edge_dst_ids, dtype=torch.long)
    edge_index = torch.stack([edge_index_src, edge_index_dst], dim=0)

    graph = PTGraph(edge_index=edge_index, num_nodes=len(node_values))
    node_values_id = torch.tensor([value2id[str(v)] for v in node_values], dtype=torch.long)
    graph.feat_id = node_values_id

    tokenizer = GraphTokenizer(graph)
    params.decoder_config.vocab_size = tokenizer.token_nums()
    params.encoder_config.graph_value_num = len(node_values_uni)
    params.device = device

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = os.path.join(params.checkpoint_path, f"{params.encoder_config.encoder_name}_checkpoint")

    return params, graph, tokenizer, dataset_name, checkpoint


def load_trained_model(params, checkpoint: str, device: torch.device) -> GraphDecoder:
    model = GraphDecoder(params).to(device)
    raw = torch.load(checkpoint, map_location=device)

    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
    else:
        state_dict = raw

    normalized = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith("module.") else k
        normalized[nk] = v

    load_res = model.load_state_dict(normalized, strict=False)
    print(f"[load] missing_keys={load_res.missing_keys}")
    print(f"[load] unexpected_keys={load_res.unexpected_keys}")

    loaded_keys = set(normalized.keys())
    if not any(k.startswith("encoder.") for k in loaded_keys):
        raise RuntimeError("Model encoder weights were not loaded. Check checkpoint/config compatibility.")
    if not any(k.startswith("decoder.") for k in loaded_keys):
        raise RuntimeError("Model decoder weights were not loaded. Check checkpoint/config compatibility.")

    model.eval()
    return model


def sample_connected_induced_query(
    data_adj: List[Set[int]],
    data_labels: torch.Tensor,
    query_size: int,
    rng: random.Random,
    max_tries: int = 200,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    n = len(data_adj)
    if query_size > n:
        raise ValueError(f"query_size={query_size} > num_nodes={n}")

    for _ in range(max_tries):
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

        q_edges = []
        for u in orig_nodes:
            for v in data_adj[u]:
                if v in selected:
                    q_edges.append((local_map[u], local_map[v]))

        if not q_edges and query_size > 1:
            continue

        q_num_nodes = len(orig_nodes)
        q_adj = [set() for _ in range(q_num_nodes)]
        for u, v in q_edges:
            q_adj[u].add(v)

        seen = {0}
        stack = [0]
        while stack:
            x = stack.pop()
            for y in q_adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        if len(seen) != q_num_nodes:
            continue

        edge_index = torch.tensor(q_edges, dtype=torch.long).t().contiguous()
        labels = data_labels[torch.tensor(orig_nodes, dtype=torch.long)].clone()
        return edge_index, labels, orig_nodes

    raise RuntimeError("Failed to sample connected induced query after many tries.")


def sample_query_stream(
    data_adj: List[Set[int]],
    data_labels: torch.Tensor,
    query_size: int,
    num_queries: int,
    seed: int,
) -> List[QuerySample]:
    rng = random.Random(seed)
    queries = []
    for qid in range(num_queries):
        q_ei, q_labels, orig_nodes = sample_connected_induced_query(data_adj, data_labels, query_size, rng)
        queries.append(QuerySample(query_id=qid, edge_index=q_ei, labels=q_labels, orig_nodes=orig_nodes))
    return queries


def save_query_stream(path: str, queries: Sequence[QuerySample]) -> None:
    records = []
    for q in queries:
        records.append(
            {
                "query_id": q.query_id,
                "edge_index": q.edge_index.tolist(),
                "labels": q.labels.tolist(),
                "orig_nodes": q.orig_nodes,
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def load_query_stream(path: str) -> List[QuerySample]:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    queries = []
    for r in records:
        queries.append(
            QuerySample(
                query_id=int(r["query_id"]),
                edge_index=torch.tensor(r["edge_index"], dtype=torch.long),
                labels=torch.tensor(r["labels"], dtype=torch.long),
                orig_nodes=[int(x) for x in r["orig_nodes"]],
            )
        )
    return queries


def build_query_ptgraph(query_edge_index: torch.Tensor, query_labels: torch.Tensor) -> PTGraph:
    q_pt = PTGraph(edge_index=query_edge_index, num_nodes=query_labels.size(0))
    q_pt.feat_id = query_labels.clone()
    return q_pt


def make_euler_masked_sequence(
    q_pt: PTGraph,
    tokenizer: GraphTokenizer,
    params,
    partial_mapping: Dict[int, int],
    next_query_node: int,
) -> Tuple[List[int], List[int], int]:
    if q_pt.num_nodes == 1:
        path_nodes = [0]
    else:
        path_edges = graph2path_v2(q_pt)
        if len(path_edges) == 0:
            path_nodes = [0]
        else:
            path_nodes = [u for u, _ in path_edges]
            path_nodes.append(path_edges[-1][1])

    subnode_id_size = params.decoder_config.sub_node_id_size
    nodeid2sub = {qnode: idx % subnode_id_size for idx, qnode in enumerate(path_nodes)}

    masked_tokens = []
    path_subnode_ids = []
    for qn in path_nodes:
        if qn == next_query_node:
            token = tokenizer.sos_id
        elif qn in partial_mapping:
            token = partial_mapping[qn]
        else:
            token = tokenizer.padding_id
        masked_tokens.append(token)
        path_subnode_ids.append(nodeid2sub[qn])

    input_seq = [tokenizer.sos_id] + masked_tokens
    subnode_seq = [nodeid2sub.get(next_query_node, 0)] + path_subnode_ids
    token_mask_len = len(input_seq)
    return input_seq, subnode_seq, token_mask_len


def score_candidates_with_neugn(
    model: GraphDecoder,
    q_edge_index: torch.Tensor,
    q_labels: torch.Tensor,
    tokenizer: GraphTokenizer,
    params,
    device: torch.device,
    partial_mapping: Dict[int, int],
    next_query_node: int,
    local_candidates: Sequence[int],
) -> List[int]:
    q_pt = build_query_ptgraph(q_edge_index, q_labels)
    batch_q = PTBatch.from_data_list([q_pt]).to(device)

    input_seq, subnode_seq, token_mask_len = make_euler_masked_sequence(
        q_pt, tokenizer, params, partial_mapping, next_query_node
    )

    input_seq_tensor = torch.tensor([input_seq], dtype=torch.long, device=device)
    subnode_seq_tensor = torch.tensor([subnode_seq], dtype=torch.long, device=device)
    token_mask_len_tensor = torch.tensor([token_mask_len], dtype=torch.long, device=device)

    with torch.no_grad():
        graph_features = model.get_encoder_tensor(batch_q, device)
        logits = model.get_decoder_output(
            graph_features,
            input_seq_tensor,
            subnode_seq_tensor,
            token_mask_len_tensor,
        )

    if logits.dim() == 3:
        logits_vec = logits[:, 0, :]
    elif logits.dim() == 2:
        logits_vec = logits
    else:
        raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

    vocab = logits_vec.size(-1)
    scores = {}
    for c in local_candidates:
        if c >= vocab:
            raise RuntimeError(
                f"Candidate id {c} out of logits vocab range {vocab}. "
                f"Possible vocab_size / graph node count mismatch."
            )
        scores[c] = float(logits_vec[0, c].item())

    ordered = sorted(local_candidates, key=lambda x: (-scores[x], x))
    return ordered


def query_order_deterministic(query_adj: List[Set[int]], query_labels: torch.Tensor) -> List[int]:
    label_freq: Dict[int, int] = {}
    for x in query_labels.tolist():
        label_freq[x] = label_freq.get(x, 0) + 1

    nodes = list(range(len(query_adj)))
    nodes.sort(key=lambda u: (-len(query_adj[u]), label_freq[int(query_labels[u])], u))
    return nodes


def is_consistent(
    qnode: int,
    dnode: int,
    mapping: Dict[int, int],
    used_data: Set[int],
    query_adj: List[Set[int]],
    data_adj: List[Set[int]],
) -> bool:
    if dnode in used_data:
        return False
    for qn in query_adj[qnode]:
        if qn in mapping:
            if mapping[qn] not in data_adj[dnode]:
                return False
    return True


def local_candidates_for_qnode(
    qnode: int,
    query_labels: torch.Tensor,
    data_labels: torch.Tensor,
    query_adj: List[Set[int]],
    data_adj: List[Set[int]],
) -> List[int]:
    q_label = int(query_labels[qnode])
    q_deg = len(query_adj[qnode])
    cands = []
    for dnode in range(len(data_adj)):
        if int(data_labels[dnode]) != q_label:
            continue
        if len(data_adj[dnode]) < q_deg:
            continue
        cands.append(dnode)
    cands.sort()
    return cands


def enumerate_first_match(
    query_order: List[int],
    query_adj: List[Set[int]],
    data_adj: List[Set[int]],
    query_labels: torch.Tensor,
    data_labels: torch.Tensor,
    candidate_order_fn,
    max_steps: Optional[int],
) -> MatchResult:
    start_t = time.perf_counter()
    mapping: Dict[int, int] = {}
    used_data: Set[int] = set()
    fms = 0

    def dfs(depth: int):
        nonlocal fms
        if max_steps is not None and fms >= max_steps:
            return None
        if depth == len(query_order):
            return dict(mapping)

        qnode = query_order[depth]
        local_cands = local_candidates_for_qnode(qnode, query_labels, data_labels, query_adj, data_adj)
        ordered_cands = candidate_order_fn(depth, qnode, local_cands, mapping)

        for dnode in ordered_cands:
            fms += 1
            if max_steps is not None and fms > max_steps:
                return None
            if not is_consistent(qnode, dnode, mapping, used_data, query_adj, data_adj):
                continue

            mapping[qnode] = dnode
            used_data.add(dnode)
            out = dfs(depth + 1)
            if out is not None:
                return out
            used_data.remove(dnode)
            del mapping[qnode]
        return None

    first_match = dfs(0)
    elapsed = time.perf_counter() - start_t
    return MatchResult(found=first_match is not None, fms=fms, first_match=first_match, elapsed_seconds=elapsed)


def enumerate_all_matches(
    query_order: List[int],
    query_adj: List[Set[int]],
    data_adj: List[Set[int]],
    query_labels: torch.Tensor,
    data_labels: torch.Tensor,
    candidate_order_fn,
    max_matches: Optional[int] = None,
    time_budget: Optional[float] = None,
) -> Tuple[List[Dict[int, int]], float]:
    start_t = time.perf_counter()
    mapping: Dict[int, int] = {}
    used_data: Set[int] = set()
    matches: List[Dict[int, int]] = []

    def should_stop() -> bool:
        if max_matches is not None and len(matches) >= max_matches:
            return True
        if time_budget is not None and (time.perf_counter() - start_t) >= time_budget:
            return True
        return False

    def dfs(depth: int):
        if should_stop():
            return
        if depth == len(query_order):
            matches.append(dict(mapping))
            return

        qnode = query_order[depth]
        local_cands = local_candidates_for_qnode(qnode, query_labels, data_labels, query_adj, data_adj)
        ordered_cands = candidate_order_fn(depth, qnode, local_cands, mapping)

        for dnode in ordered_cands:
            if should_stop():
                return
            if not is_consistent(qnode, dnode, mapping, used_data, query_adj, data_adj):
                continue

            mapping[qnode] = dnode
            used_data.add(dnode)
            dfs(depth + 1)
            used_data.remove(dnode)
            del mapping[qnode]

    dfs(0)
    elapsed = time.perf_counter() - start_t
    return matches, elapsed


def enumerate_time_budget(
    query_order: List[int],
    query_adj: List[Set[int]],
    data_adj: List[Set[int]],
    query_labels: torch.Tensor,
    data_labels: torch.Tensor,
    candidate_order_fn,
    time_budget: float,
    max_matches: Optional[int],
) -> Tuple[int, float, float]:
    matches, elapsed = enumerate_all_matches(
        query_order,
        query_adj,
        data_adj,
        query_labels,
        data_labels,
        candidate_order_fn,
        max_matches=max_matches,
        time_budget=time_budget,
    )
    elapsed = max(elapsed, 1e-9)
    mps = len(matches) / elapsed
    return len(matches), elapsed, mps


def summarize_results(rows: List[dict], has_mps: bool, completeness_pass: Optional[bool]) -> None:
    def valid(vals):
        return [v for v in vals if v is not None]

    base_fms = valid([r["baseline_fms"] for r in rows])
    neug_fms = valid([r["neugn_fms"] for r in rows])
    improv = valid([r["improvement_percent"] for r in rows])

    print("\n===== Summary =====")
    if base_fms:
        print(f"All median baseline_fms: {median(base_fms):.3f}")
        print(f"All median neugn_fms: {median(neug_fms):.3f}")
    if improv:
        print(f"All median improvement(%): {median(improv):.3f}")

    for group in ["dense", "sparse"]:
        g = [r for r in rows if r["density_group"] == group]
        if not g:
            continue
        gb = valid([r["baseline_fms"] for r in g])
        gn = valid([r["neugn_fms"] for r in g])
        gi = valid([r["improvement_percent"] for r in g])
        if gb:
            print(
                f"{group} median baseline/neugn/improvement: "
                f"{median(gb):.3f} / {median(gn):.3f} / {median(gi):.3f}%"
            )

    if has_mps:
        b_mps = valid([r["baseline_mps"] for r in rows])
        n_mps = valid([r["neugn_mps"] for r in rows])
        if b_mps:
            print(f"baseline MPS mean/median: {mean(b_mps):.3f} / {median(b_mps):.3f}")
            print(f"neugn MPS mean/median: {mean(n_mps):.3f} / {median(n_mps):.3f}")

    if completeness_pass is not None:
        print(f"completeness check: {'PASS' if completeness_pass else 'FAIL'}")


def canonicalize_match(m: Dict[int, int]) -> Tuple[Tuple[int, int], ...]:
    return tuple(sorted(m.items(), key=lambda x: x[0]))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params, graph, tokenizer, dataset_name, checkpoint = load_data_graph(args, device)
    print(f"[info] dataset={dataset_name} graph_nodes={graph.num_nodes} checkpoint={checkpoint}")

    model = load_trained_model(params, checkpoint, device)

    data_adj = build_adjacency(graph.num_nodes, graph.edge_index)
    data_labels = graph.feat_id.clone()

    if args.load_queries:
        query_stream = load_query_stream(args.load_queries)
        print(f"[info] loaded {len(query_stream)} queries from {args.load_queries}")
    else:
        query_stream = sample_query_stream(data_adj, data_labels, args.query_size, args.num_queries, args.seed)
        if args.save_queries:
            save_query_stream(args.save_queries, query_stream)
            print(f"[info] saved query stream to {args.save_queries}")

    rows = []
    completeness_pass: Optional[bool] = None

    for q in query_stream:
        q_n = int(q.labels.numel())
        q_adj = build_adjacency(q_n, q.edge_index)
        q_order = query_order_deterministic(q_adj, q.labels)

        q_edges = sum(len(s) for s in q_adj) / 2.0
        q_avg_degree = (2.0 * q_edges) / max(q_n, 1)
        density_group = "dense" if q_avg_degree >= 3.0 else "sparse"

        def baseline_candidate_order_fn(depth, qnode, local_cands, mapping):
            return sorted(local_cands)

        def neugn_candidate_order_fn(depth, qnode, local_cands, mapping):
            if depth >= args.nav_depth:
                return sorted(local_cands)
            return score_candidates_with_neugn(
                model=model,
                q_edge_index=q.edge_index,
                q_labels=q.labels,
                tokenizer=tokenizer,
                params=params,
                device=device,
                partial_mapping=mapping,
                next_query_node=qnode,
                local_candidates=local_cands,
            )

        base_first = enumerate_first_match(
            q_order, q_adj, data_adj, q.labels, data_labels,
            baseline_candidate_order_fn, args.max_steps
        )
        neug_first = enumerate_first_match(
            q_order, q_adj, data_adj, q.labels, data_labels,
            neugn_candidate_order_fn, args.max_steps
        )

        improvement = None
        if base_first.found and neug_first.found and base_first.fms > 0:
            improvement = (base_first.fms - neug_first.fms) / base_first.fms * 100.0

        baseline_mps = None
        neugn_mps = None
        if args.time_budget is not None:
            _, _, baseline_mps = enumerate_time_budget(
                q_order, q_adj, data_adj, q.labels, data_labels,
                baseline_candidate_order_fn,
                args.time_budget,
                args.max_matches,
            )
            _, _, neugn_mps = enumerate_time_budget(
                q_order, q_adj, data_adj, q.labels, data_labels,
                neugn_candidate_order_fn,
                args.time_budget,
                args.max_matches,
            )

        row = {
            "query_id": q.query_id,
            "query_size": q_n,
            "query_edges": int(q_edges),
            "query_avg_degree": q_avg_degree,
            "density_group": density_group,
            "baseline_fms": base_first.fms,
            "neugn_fms": neug_first.fms,
            "improvement_percent": improvement,
            "baseline_time": base_first.elapsed_seconds,
            "neugn_time": neug_first.elapsed_seconds,
            "baseline_found": base_first.found,
            "neugn_found": neug_first.found,
            "baseline_mps": baseline_mps,
            "neugn_mps": neugn_mps,
        }
        rows.append(row)

        print(
            f"[query {q.query_id}] size={q_n} group={density_group} "
            f"baseline_fms={base_first.fms} neugn_fms={neug_first.fms} "
            f"improve={improvement if improvement is not None else 'N/A'}"
        )

        if args.check_completeness:
            if q_n > 10 and args.max_matches is None:
                raise ValueError("--check_completeness requires query_size<=10 or --max_matches set")

            base_matches, _ = enumerate_all_matches(
                q_order, q_adj, data_adj, q.labels, data_labels,
                baseline_candidate_order_fn,
                max_matches=args.max_matches,
            )
            neug_matches, _ = enumerate_all_matches(
                q_order, q_adj, data_adj, q.labels, data_labels,
                neugn_candidate_order_fn,
                max_matches=args.max_matches,
            )

            base_set = {canonicalize_match(m) for m in base_matches}
            neug_set = {canonicalize_match(m) for m in neug_matches}
            if base_set != neug_set:
                only_base = next(iter(base_set - neug_set), None)
                only_neug = next(iter(neug_set - base_set), None)
                print(f"[completeness] first diff only_base={only_base} only_neugn={only_neug}")
                raise AssertionError("Completeness check failed: baseline and NeuGN match sets differ.")
            completeness_pass = True
            print(f"[completeness] query {q.query_id}: PASS")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fieldnames = [
        "query_id",
        "query_size",
        "query_edges",
        "query_avg_degree",
        "density_group",
        "baseline_fms",
        "neugn_fms",
        "improvement_percent",
        "baseline_time",
        "neugn_time",
        "baseline_found",
        "neugn_found",
        "baseline_mps",
        "neugn_mps",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] wrote {len(rows)} rows to {args.output}")

    summarize_results(rows, has_mps=args.time_budget is not None, completeness_pass=completeness_pass)


if __name__ == "__main__":
    main()

# Example:
# python method/demo.py \
#   --config_path ./model_params/wikics \
#   --graph_path ../datasets/wikics \
#   --dataset wikics \
#   --query_size 20 \
#   --num_queries 200 \
#   --nav_depth 10 \
#   --device cuda \
#   --output ./demo_results_wikics.csv
#
# MPS example:
# python method/demo.py \
#   --config_path ./model_params/wikics \
#   --graph_path ../datasets/wikics \
#   --dataset wikics \
#   --query_size 32 \
#   --num_queries 200 \
#   --nav_depth 16 \
#   --time_budget 1.0 \
#   --device cuda \
#   --output ./demo_mps_wikics.csv
