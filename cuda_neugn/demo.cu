#include "include/device_graph.cuh"
#include "include/device_neugn.cuh"
#include "include/gpu_matcher.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct DemoArgs {
    std::string export_dir = "./cuda_export/wikics";
    std::string config_path = "./method/model_params/wikics";
    std::string graph_path = "./datasets/wikics";
    std::string dataset = "wikics";
    std::string query_bin = "";
    std::string output = "./demo_cu_fused_results.csv";
    int query_size = 20;
    int num_queries = 200;
    int nav_depth = 10;
    std::optional<double> time_budget;
    std::optional<int> max_steps;
    std::optional<int> max_matches;
    bool check_completeness = false;
    int seed = 42;
    int device_id = 0;
    int print_first = 5;
};

struct Query {
    int query_id = 0;
    int n = 0;
    std::vector<int> edge_src;
    std::vector<int> edge_dst;
    std::vector<int> labels;
};

struct CsvRow {
    int query_id = 0;
    int query_size = 0;
    int query_edges = 0;
    double query_avg_degree = 0.0;
    std::string density_group;
    int baseline_fms = 0;
    int neugn_fms = 0;
    std::optional<double> improvement_percent;
    double baseline_time = 0.0;
    double neugn_time = 0.0;
    bool baseline_found = false;
    bool neugn_found = false;
    std::optional<double> baseline_mps;
    std::optional<double> neugn_mps;
};

void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
}

DemoArgs parse_args(int argc, char** argv) {
    DemoArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + a);
            return std::string(argv[++i]);
        };
        if (a == "--export_dir") args.export_dir = next();
        else if (a == "--config_path") args.config_path = next();
        else if (a == "--graph_path") args.graph_path = next();
        else if (a == "--dataset") args.dataset = next();
        else if (a == "--query_bin") args.query_bin = next();
        else if (a == "--output") args.output = next();
        else if (a == "--query_size") args.query_size = std::stoi(next());
        else if (a == "--num_queries") args.num_queries = std::stoi(next());
        else if (a == "--nav_depth") args.nav_depth = std::stoi(next());
        else if (a == "--time_budget") args.time_budget = std::stod(next());
        else if (a == "--max_steps") args.max_steps = std::stoi(next());
        else if (a == "--max_matches") args.max_matches = std::stoi(next());
        else if (a == "--check_completeness") args.check_completeness = true;
        else if (a == "--seed") args.seed = std::stoi(next());
        else if (a == "--device_id") args.device_id = std::stoi(next());
        else if (a == "--print_first") args.print_first = std::stoi(next());
        else throw std::runtime_error("Unknown argument: " + a);
    }
    if (args.query_bin.empty()) args.query_bin = args.export_dir + "/demo_input/queries.bin";
    return args;
}


std::unordered_map<int,int> load_value2id_csv(const std::string& config_path, const std::string& dataset) {
    std::string path = config_path + "/" + dataset + "_value2id_mapping.csv";
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open value2id csv: " + path);
    std::unordered_map<int,int> out;
    std::string line;
    std::getline(in, line); // header
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto c = line.find(',');
        if (c == std::string::npos) continue;
        int val = std::stoi(line.substr(0, c));
        int id = std::stoi(line.substr(c + 1));
        out[val] = id;
    }
    return out;
}

void load_data_graph_from_text(
    const DemoArgs& args,
    std::vector<int>& data_src,
    std::vector<int>& data_dst,
    std::vector<int>& data_labels
) {
    const std::string path = args.graph_path + "/" + args.dataset + ".graph";
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open graph file: " + path);

    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("Empty .graph file: " + path);
    std::stringstream hs(line);
    char ttag = 0;
    int n_nodes = 0, n_edges = 0;
    hs >> ttag;
    if (ttag != 't') throw std::runtime_error("Invalid .graph header: " + path);
    // Support both: "t <num_nodes> <num_edges>" and "t <graph_id> <num_nodes> <num_edges>"
    std::vector<int> hnums;
    int x = 0;
    while (hs >> x) hnums.push_back(x);
    if (hnums.size() == 2) {
        n_nodes = hnums[0];
        n_edges = hnums[1];
    } else if (hnums.size() == 3) {
        n_nodes = hnums[1];
        n_edges = hnums[2];
    } else {
        throw std::runtime_error("Unsupported .graph header format: " + path);
    }

    data_labels.assign(n_nodes, 0);
    auto value2id = load_value2id_csv(args.config_path, args.dataset);

    int v_read = 0;
    int e_read = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        char tag = 0;
        ss >> tag;
        if (tag == 'v') {
            int nid = 0;
            int raw_label = 0;
            ss >> nid >> raw_label;
            if (!ss) throw std::runtime_error("Invalid vertex row in .graph: " + path);
            auto it = value2id.find(raw_label);
            if (it == value2id.end()) throw std::runtime_error("Missing label in value2id mapping: " + std::to_string(raw_label));
            if (nid < 0 || nid >= n_nodes) throw std::runtime_error("Node id out of range in .graph");
            data_labels[nid] = it->second;
            ++v_read;
        } else if (tag == 'e') {
            int s = 0, d = 0;
            ss >> s >> d;
            if (!ss) throw std::runtime_error("Invalid edge row in .graph: " + path);
            data_src.push_back(s);
            data_dst.push_back(d);
            data_src.push_back(d);
            data_dst.push_back(s);
            ++e_read;
        }
    }

    if (v_read != n_nodes) throw std::runtime_error("Vertex count mismatch when reading .graph");
    if (e_read != n_edges) throw std::runtime_error("Edge count mismatch when reading .graph");
}

std::vector<int> read_i32_bin(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open int32 binary: " + path);
    in.seekg(0, std::ios::end);
    auto nbytes = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    if (nbytes % sizeof(int32_t) != 0) throw std::runtime_error("Invalid int32 binary size: " + path);
    std::vector<int32_t> tmp(nbytes / sizeof(int32_t));
    in.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(nbytes));
    return std::vector<int>(tmp.begin(), tmp.end());
}

std::vector<Query> load_queries_bin(const std::string& path, int limit) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open query bin: " + path);
    int32_t qn = 0;
    in.read(reinterpret_cast<char*>(&qn), sizeof(int32_t));
    if (!in) throw std::runtime_error("Failed to read query count");

    std::vector<Query> out;
    out.reserve(std::min(limit, static_cast<int>(qn)));
    for (int i = 0; i < qn && static_cast<int>(out.size()) < limit; ++i) {
        int32_t qid = 0, n = 0, e = 0, orig_n = 0;
        in.read(reinterpret_cast<char*>(&qid), sizeof(int32_t));
        in.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
        in.read(reinterpret_cast<char*>(&e), sizeof(int32_t));
        in.read(reinterpret_cast<char*>(&orig_n), sizeof(int32_t));

        Query q;
        q.query_id = qid;
        q.n = n;

        std::vector<int32_t> labels(n), edges(2 * e), orig(orig_n);
        in.read(reinterpret_cast<char*>(labels.data()), static_cast<std::streamsize>(labels.size() * sizeof(int32_t)));
        in.read(reinterpret_cast<char*>(edges.data()), static_cast<std::streamsize>(edges.size() * sizeof(int32_t)));
        in.read(reinterpret_cast<char*>(orig.data()), static_cast<std::streamsize>(orig.size() * sizeof(int32_t)));
        if (!in) throw std::runtime_error("Corrupt query bin payload");

        q.labels.assign(labels.begin(), labels.end());
        q.edge_src.resize(e);
        q.edge_dst.resize(e);
        for (int k = 0; k < e; ++k) {
            q.edge_src[k] = edges[k];
            q.edge_dst[k] = edges[e + k];
        }
        out.push_back(std::move(q));
    }
    return out;
}

std::unordered_map<int, std::vector<int>> load_query_paths_bin(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    int32_t qn = 0;
    in.read(reinterpret_cast<char*>(&qn), sizeof(int32_t));
    if (!in) return {};
    std::unordered_map<int, std::vector<int>> out;
    for (int i = 0; i < qn; ++i) {
        int32_t qid = 0, len = 0;
        in.read(reinterpret_cast<char*>(&qid), sizeof(int32_t));
        in.read(reinterpret_cast<char*>(&len), sizeof(int32_t));
        if (!in) break;
        std::vector<int32_t> nodes(len);
        in.read(reinterpret_cast<char*>(nodes.data()), static_cast<std::streamsize>(len * sizeof(int32_t)));
        if (!in) break;
        out[qid] = std::vector<int>(nodes.begin(), nodes.end());
    }
    return out;
}

HostQueryBatch build_host_query_batch(const std::vector<Query>& queries, const std::unordered_map<int, std::vector<int>>& paths) {
    HostQueryBatch out;
    out.num_queries = static_cast<int>(queries.size());
    out.q_num_nodes.resize(out.num_queries, 0);
    out.q_num_edges.resize(out.num_queries, 0);
    out.q_labels.assign(out.num_queries * out.max_query_nodes, 0);
    out.q_adj_row_ptr.assign(out.num_queries * (out.max_query_nodes + 1), 0);
    out.q_adj_col_idx.assign(out.num_queries * out.max_query_edges, 0);
    out.q_degree.assign(out.num_queries * out.max_query_nodes, 0);
    out.q_path_ptr.assign(out.num_queries + 1, 0);

    for (int qid = 0; qid < out.num_queries; ++qid) {
        const Query& q = queries[qid];
        if (q.n > out.max_query_nodes) throw std::runtime_error("query size exceeds MAX_QUERY_NODES");

        out.q_num_nodes[qid] = q.n;
        out.q_num_edges[qid] = static_cast<int>(q.edge_src.size());

        int lbase = qid * out.max_query_nodes;
        for (int i = 0; i < q.n; ++i) out.q_labels[lbase + i] = q.labels[i];

        std::vector<std::vector<int>> adj(q.n);
        for (size_t e = 0; e < q.edge_src.size(); ++e) {
            int s = q.edge_src[e], d = q.edge_dst[e];
            if (s < 0 || s >= q.n || d < 0 || d >= q.n || s == d) continue;
            adj[s].push_back(d);
            adj[d].push_back(s);
        }
        for (auto& nbrs : adj) {
            std::sort(nbrs.begin(), nbrs.end());
            nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
        }

        int rpbase = qid * (out.max_query_nodes + 1);
        int cbase = qid * out.max_query_edges;
        int c = 0;
        out.q_adj_row_ptr[rpbase] = 0;
        for (int u = 0; u < q.n; ++u) {
            out.q_degree[lbase + u] = static_cast<int>(adj[u].size());
            for (int v : adj[u]) {
                if (c >= out.max_query_edges) break;
                out.q_adj_col_idx[cbase + c++] = v;
            }
            out.q_adj_row_ptr[rpbase + u + 1] = c;
        }
        for (int u = q.n; u < out.max_query_nodes; ++u) out.q_adj_row_ptr[rpbase + u + 1] = c;

        auto it = paths.find(q.query_id);
        std::vector<int> p;
        if (it != paths.end()) p = it->second;
        else {
            p.resize(q.n);
            std::iota(p.begin(), p.end(), 0);
        }
        out.q_path_ptr[qid + 1] = out.q_path_ptr[qid] + static_cast<int>(p.size());
        out.q_path_nodes.insert(out.q_path_nodes.end(), p.begin(), p.end());
    }

    return out;
}

std::string density_group(int n, int e) {
    if (n <= 1) return "sparse";
    double den = (2.0 * e) / (n * (n - 1));
    if (den < 0.33) return "sparse";
    if (den < 0.66) return "medium";
    return "dense";
}

void write_results_csv(const std::string& path, const std::vector<CsvRow>& rows) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write CSV: " + path);
    out << "query_id,query_size,query_edges,query_avg_degree,density_group,baseline_fms,neugn_fms,improvement_percent,baseline_time,neugn_time,baseline_found,neugn_found,baseline_mps,neugn_mps\n";
    for (const auto& r : rows) {
        out << r.query_id << ',' << r.query_size << ',' << r.query_edges << ',' << r.query_avg_degree << ','
            << r.density_group << ',' << r.baseline_fms << ',' << r.neugn_fms << ',';
        if (r.improvement_percent) out << *r.improvement_percent;
        out << ',' << r.baseline_time << ',' << r.neugn_time << ',' << (r.baseline_found ? 1 : 0) << ',' << (r.neugn_found ? 1 : 0) << ',';
        if (r.baseline_mps) out << *r.baseline_mps;
        out << ',';
        if (r.neugn_mps) out << *r.neugn_mps;
        out << '\n';
    }
}

float event_elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    return ms;
}

} // namespace

int main(int argc, char** argv) {
    DemoArgs args = parse_args(argc, argv);
    cuda_check(cudaSetDevice(args.device_id), "cudaSetDevice");

    std::vector<int> data_src, data_dst, data_labels;
    load_data_graph_from_text(args, data_src, data_dst, data_labels);

    auto queries = load_queries_bin(args.query_bin, args.num_queries);
    auto query_paths = load_query_paths_bin(args.export_dir + "/demo_input/query_paths.bin");
    HostQueryBatch h_queries = build_host_query_batch(queries, query_paths);

    DeviceGraphCSR d_graph = upload_graph_csr(static_cast<int>(data_labels.size()), data_src, data_dst, data_labels);
    DeviceQueryBatch d_queries = upload_query_batch(h_queries);

    DeviceMatcherConfig cfg;
    cfg.nav_depth = args.nav_depth;
    if (args.max_steps) cfg.max_steps = *args.max_steps;
    if (args.max_matches) cfg.max_matches = *args.max_matches;

    DeviceNeuGNWeightsOwner weights = load_device_neugn_weights(args.export_dir, cfg);

    DeviceCandidateBuffer d_cands;
    alloc_candidate_buffer(d_queries.num_queries, d_queries.max_query_nodes, d_graph.num_nodes, d_cands);

    DeviceMatchResult* d_results = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_results), d_queries.num_queries * sizeof(DeviceMatchResult)), "cudaMalloc results");
    cuda_check(cudaMemset(d_results, 0, d_queries.num_queries * sizeof(DeviceMatchResult)), "cudaMemset results");

    cudaEvent_t e0, e1, e2, e3, e4;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    cudaEventCreate(&e3);
    cudaEventCreate(&e4);

    cudaEventRecord(e0);
    launch_build_initial_candidates(d_graph, d_queries, cfg, d_cands, d_results);
    cudaEventRecord(e1);

    launch_build_query_order(d_graph, d_queries);
    cudaEventRecord(e2);

    launch_gpu_baseline_join(d_graph, d_queries, cfg, d_cands, d_results);
    cudaEventRecord(e3);

    launch_gpu_neugn_fused_join(d_graph, d_queries, cfg, d_cands, weights.device_view, d_results);
    cudaEventRecord(e4);

    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::vector<DeviceMatchResult> results(d_queries.num_queries);
    cuda_check(cudaMemcpy(results.data(), d_results, results.size() * sizeof(DeviceMatchResult), cudaMemcpyDeviceToHost), "cudaMemcpy results D2H");

    float filter_ms = event_elapsed_ms(e0, e1);
    float order_ms = event_elapsed_ms(e1, e2);
    float baseline_join_ms = event_elapsed_ms(e2, e3);
    float neugn_join_ms = event_elapsed_ms(e3, e4);

    std::vector<CsvRow> rows;
    rows.reserve(queries.size());
    for (size_t i = 0; i < queries.size(); ++i) {
        CsvRow row;
        row.query_id = queries[i].query_id;
        row.query_size = queries[i].n;
        row.query_edges = static_cast<int>(queries[i].edge_src.size());
        row.query_avg_degree = (row.query_size > 0) ? (2.0 * row.query_edges / row.query_size) : 0.0;
        row.density_group = density_group(row.query_size, row.query_edges);
        row.baseline_fms = results[i].baseline_fms;
        row.neugn_fms = results[i].neugn_fms;
        row.baseline_time = baseline_join_ms / queries.size();
        row.neugn_time = neugn_join_ms / queries.size();
        row.baseline_found = results[i].baseline_found;
        row.neugn_found = results[i].neugn_found;
        if (row.baseline_fms > 0) {
            row.improvement_percent = 100.0 * (static_cast<double>(row.baseline_fms - row.neugn_fms) / row.baseline_fms);
        }
        if (row.baseline_time > 0.0) row.baseline_mps = static_cast<double>(row.baseline_found) / (row.baseline_time / 1000.0);
        if (row.neugn_time > 0.0) row.neugn_mps = static_cast<double>(row.neugn_found) / (row.neugn_time / 1000.0);
        rows.push_back(row);
    }

    write_results_csv(args.output, rows);
    std::cout << "filter_ms=" << filter_ms
              << " order_ms=" << order_ms
              << " baseline_join_ms=" << baseline_join_ms
              << " neugn_join_ms=" << neugn_join_ms << "\n";

    cudaFree(d_results);
    free_candidate_buffer(d_cands);
    free_device_neugn_weights(weights);
    free_query_batch(d_queries);
    free_graph_csr(d_graph);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
    cudaEventDestroy(e3);
    cudaEventDestroy(e4);
    return 0;
}
