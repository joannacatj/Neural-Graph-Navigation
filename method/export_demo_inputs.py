import argparse
import os
import struct
import subprocess
import sys
import torch

from demo import (
    build_adjacency,
    load_data_graph,
    load_trained_model,
    sample_query_stream,
    save_query_stream,
)
from NeuGN.nx_utils import graph2path_v2
from NeuGN.pt_graph import PTGraph


def parse_args():
    p = argparse.ArgumentParser(description="Export demo inputs for CUDA demo_cu.")
    p.add_argument("--config_path", default="./model_params/wikics", type=str)
    p.add_argument("--checkpoint", default=None, type=str)
    p.add_argument("--graph_path", default=None, type=str)
    p.add_argument("--dataset", default=None, type=str)
    p.add_argument("--out_dir", default="../cuda_export/wikics", type=str)
    p.add_argument("--query_size", default=20, type=int)
    p.add_argument("--num_queries", default=200, type=int)
    p.add_argument("--nav_depth", default=10, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--device", default=None, type=str)
    p.add_argument("--output_python_csv", default="../cuda_export/wikics/demo_py_results.csv", type=str)
    return p.parse_args()


def write_i32_bin(path, arr):
    t = torch.tensor(arr, dtype=torch.int32).contiguous().cpu().numpy()
    t.tofile(path)


def export_query_stream_bin(path, queries):
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(queries)))
        for q in queries:
            n = int(q.labels.numel())
            e = int(q.edge_index.size(1))
            orig_n = len(q.orig_nodes)
            f.write(struct.pack("<4i", int(q.query_id), n, e, orig_n))
            f.write(torch.tensor(q.labels.tolist(), dtype=torch.int32).numpy().tobytes())
            f.write(torch.tensor(q.edge_index.reshape(-1).tolist(), dtype=torch.int32).numpy().tobytes())
            f.write(torch.tensor(q.orig_nodes, dtype=torch.int32).numpy().tobytes())


def export_query_paths(path, queries):
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(queries)))
        for q in queries:
            g = PTGraph(edge_index=q.edge_index, num_nodes=q.labels.numel())
            if g.num_nodes == 1:
                path_nodes = [0]
            else:
                path_edges = graph2path_v2(g)
                path_nodes = [0] if len(path_edges) == 0 else [u for u, _ in path_edges] + [path_edges[-1][1]]
            f.write(struct.pack("<2i", int(q.query_id), len(path_nodes)))
            f.write(torch.tensor(path_nodes, dtype=torch.int32).numpy().tobytes())


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    demo_input = os.path.join(args.out_dir, "demo_input")
    os.makedirs(demo_input, exist_ok=True)

    class A:
        pass

    dargs = A()
    dargs.config_path = args.config_path
    dargs.checkpoint = args.checkpoint
    dargs.graph_path = args.graph_path
    dargs.dataset = args.dataset
    dargs.query_size = args.query_size
    dargs.num_queries = args.num_queries
    dargs.seed = args.seed

    params, graph, tokenizer, dataset_name, checkpoint = load_data_graph(dargs, device)
    _model = load_trained_model(params, checkpoint, device)
    data_adj = build_adjacency(graph.num_nodes, graph.edge_index)
    queries = sample_query_stream(data_adj, graph.feat_id, args.query_size, args.num_queries, args.seed)

    with open(os.path.join(demo_input, "data_num_nodes.txt"), "w", encoding="utf-8") as f:
        f.write(str(int(graph.num_nodes)))
    write_i32_bin(os.path.join(demo_input, "data_edges_i32.bin"), graph.edge_index.reshape(-1).tolist())
    write_i32_bin(os.path.join(demo_input, "data_labels_i32.bin"), graph.feat_id.tolist())
    export_query_stream_bin(os.path.join(demo_input, "queries.bin"), queries)
    export_query_paths(os.path.join(demo_input, "query_paths.bin"), queries)

    with open(os.path.join(demo_input, "tokenizer_meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"sos_id={tokenizer.sos_id}\n")
        f.write(f"padding_id={tokenizer.padding_id}\n")
        f.write(f"sub_node_id_size={params.decoder_config.sub_node_id_size}\n")

    query_json = os.path.join(demo_input, "queries.json")
    save_query_stream(query_json, queries)

    cmd = [
        sys.executable, "demo.py",
        "--config_path", args.config_path,
        "--graph_path", args.graph_path if args.graph_path else "",
        "--dataset", dataset_name,
        "--query_size", str(args.query_size),
        "--num_queries", str(args.num_queries),
        "--nav_depth", str(args.nav_depth),
        "--seed", str(args.seed),
        "--device", str(device),
        "--load_queries", query_json,
        "--output", args.output_python_csv,
    ]
    cmd = [x for x in cmd if x != ""]
    subprocess.run(cmd, check=True, cwd=os.path.dirname(__file__))
    print(f"[done] demo inputs exported to {demo_input}")
    print(f"[done] python demo csv: {args.output_python_csv}")


if __name__ == "__main__":
    main()
