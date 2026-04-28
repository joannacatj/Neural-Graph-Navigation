#include "include/neug_model.hpp"
#include "include/tensor_io.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define CHECK_CUDA(expr)                                                                 \
    do {                                                                                 \
        cudaError_t _err = (expr);                                                       \
        if (_err != cudaSuccess) {                                                       \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err)); \
        }                                                                                \
    } while (0)

namespace {
struct DemoArgs {
    std::string export_dir = "./cuda_export/wikics";
    std::string graph_bin = "";
    std::string query_bin = "";
    std::string output = "./demo_cu_results.csv";
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
    std::vector<int> orig_nodes;
};

struct MatchResult {
    bool found = false;
    int fms = 0;
    double elapsed = 0.0;
    std::unordered_map<int, int> first_match;
};

struct TokenizerMeta {
    int sos_id = -1;
    int padding_id = -1;
    int sub_node_id_size = 32;
};

struct Row {
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

DemoArgs parse_args(int argc, char** argv) {
    DemoArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
            return std::string(argv[++i]);
        };
        if (a == "--export_dir") args.export_dir = next(a);
        else if (a == "--graph_bin") args.graph_bin = next(a);
        else if (a == "--query_bin") args.query_bin = next(a);
        else if (a == "--output") args.output = next(a);
        else if (a == "--query_size") args.query_size = std::stoi(next(a));
        else if (a == "--num_queries") args.num_queries = std::stoi(next(a));
        else if (a == "--nav_depth") args.nav_depth = std::stoi(next(a));
        else if (a == "--time_budget") args.time_budget = std::stod(next(a));
        else if (a == "--max_steps") args.max_steps = std::stoi(next(a));
        else if (a == "--max_matches") args.max_matches = std::stoi(next(a));
        else if (a == "--check_completeness") args.check_completeness = true;
        else if (a == "--seed") args.seed = std::stoi(next(a));
        else if (a == "--device_id") args.device_id = std::stoi(next(a));
        else if (a == "--print_first") args.print_first = std::stoi(next(a));
        else throw std::runtime_error("Unknown argument: " + a);
    }
    if (args.graph_bin.empty()) args.graph_bin = args.export_dir + "/demo_input/data_edges_i32.bin";
    if (args.query_bin.empty()) args.query_bin = args.export_dir + "/demo_input/queries.bin";
    return args;
}

TokenizerMeta load_tokenizer_meta(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open tokenizer meta: " + path);
    TokenizerMeta m;
    std::string line;
    while (std::getline(in, line)) {
        auto p = line.find('=');
        if (p == std::string::npos) continue;
        auto k = line.substr(0, p);
        auto v = line.substr(p + 1);
        if (k == "sos_id") m.sos_id = std::stoi(v);
        else if (k == "padding_id") m.padding_id = std::stoi(v);
        else if (k == "sub_node_id_size") m.sub_node_id_size = std::stoi(v);
    }
    if (m.sos_id < 0 || m.padding_id < 0) throw std::runtime_error("Invalid tokenizer meta values");
    return m;
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
    std::vector<int> out(tmp.begin(), tmp.end());
    return out;
}

void write_i64_bin(const std::string& path, const std::vector<int64_t>& data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open for write: " + path);
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(int64_t)));
}

std::vector<std::unordered_set<int>> build_adj(int n, const std::vector<int>& src, const std::vector<int>& dst) {
    std::vector<std::unordered_set<int>> adj(n);
    for (size_t i = 0; i < src.size(); ++i) {
        int s = src[i], d = dst[i];
        if (s >= 0 && s < n && d >= 0 && d < n && s != d) adj[s].insert(d);
    }
    return adj;
}

std::vector<Query> load_queries_bin(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open query bin: " + path);
    int32_t qn = 0;
    in.read(reinterpret_cast<char*>(&qn), sizeof(int32_t));
    if (!in) throw std::runtime_error("Failed to read query count from " + path);
    std::vector<Query> out;
    out.reserve(qn);
    for (int i = 0; i < qn; ++i) {
        int32_t qid, n, e, orig_n;
        in.read(reinterpret_cast<char*>(&qid), sizeof(int32_t));
        in.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
        in.read(reinterpret_cast<char*>(&e), sizeof(int32_t));
        in.read(reinterpret_cast<char*>(&orig_n), sizeof(int32_t));
        if (!in) throw std::runtime_error("Corrupt query header in " + path);
        Query q;
        q.query_id = qid;
        q.n = n;
        std::vector<int32_t> labels(n), edges(2 * e), orig(orig_n);
        in.read(reinterpret_cast<char*>(labels.data()), static_cast<std::streamsize>(labels.size() * sizeof(int32_t)));
        in.read(reinterpret_cast<char*>(edges.data()), static_cast<std::streamsize>(edges.size() * sizeof(int32_t)));
        in.read(reinterpret_cast<char*>(orig.data()), static_cast<std::streamsize>(orig.size() * sizeof(int32_t)));
        if (!in) throw std::runtime_error("Corrupt query payload in " + path);
        q.labels.assign(labels.begin(), labels.end());
        q.edge_src.resize(e);
        q.edge_dst.resize(e);
        for (int k = 0; k < e; ++k) q.edge_src[k] = edges[k];
        for (int k = 0; k < e; ++k) q.edge_dst[k] = edges[e + k];
        q.orig_nodes.assign(orig.begin(), orig.end());
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
    std::unordered_map<int, std::vector<int>> m;
    for (int i = 0; i < qn; ++i) {
        int32_t qid, len;
        in.read(reinterpret_cast<char*>(&qid), sizeof(int32_t));
        in.read(reinterpret_cast<char*>(&len), sizeof(int32_t));
        if (!in) break;
        std::vector<int32_t> nodes(len);
        in.read(reinterpret_cast<char*>(nodes.data()), static_cast<std::streamsize>(len * sizeof(int32_t)));
        if (!in) break;
        m[qid] = std::vector<int>(nodes.begin(), nodes.end());
    }
    return m;
}

std::vector<int> query_order(const Query& q, const std::vector<std::unordered_set<int>>& q_adj) {
    std::unordered_map<int, int> freq;
    for (int x : q.labels) freq[x]++;
    std::vector<int> nodes(q.n);
    std::iota(nodes.begin(), nodes.end(), 0);
    std::sort(nodes.begin(), nodes.end(), [&](int a, int b) {
        int da = static_cast<int>(q_adj[a].size()), db = static_cast<int>(q_adj[b].size());
        if (da != db) return da > db;
        int fa = freq[q.labels[a]], fb = freq[q.labels[b]];
        if (fa != fb) return fa < fb;
        return a < b;
    });
    return nodes;
}

bool consistent(
    int qnode,
    int dnode,
    const std::unordered_map<int, int>& mapping,
    const std::unordered_set<int>& used_data,
    const std::vector<std::unordered_set<int>>& q_adj,
    const std::vector<std::unordered_set<int>>& d_adj
) {
    if (used_data.count(dnode)) return false;
    for (int nbr : q_adj[qnode]) {
        auto it = mapping.find(nbr);
        if (it != mapping.end()) {
            if (!d_adj[dnode].count(it->second)) return false;
        }
    }
    return true;
}

std::vector<int> local_candidates(
    int qnode,
    const Query& q,
    const std::vector<int>& data_labels,
    const std::vector<std::unordered_set<int>>& q_adj,
    const std::vector<std::unordered_set<int>>& d_adj
) {
    std::vector<int> c;
    for (int d = 0; d < static_cast<int>(d_adj.size()); ++d) {
        if (data_labels[d] != q.labels[qnode]) continue;
        if (static_cast<int>(d_adj[d].size()) < static_cast<int>(q_adj[qnode].size())) continue;
        c.push_back(d);
    }
    std::sort(c.begin(), c.end());
    return c;
}

std::vector<int> build_path_fallback(const Query& q, const std::vector<std::unordered_set<int>>& q_adj) {
    if (q.n == 1) return {0};
    std::vector<int> nodes;
    std::vector<int> seen(q.n, 0);
    std::vector<int> stack = {0};
    while (!stack.empty()) {
        int u = stack.back();
        stack.pop_back();
        if (seen[u]) continue;
        seen[u] = 1;
        nodes.push_back(u);
        std::vector<int> nbrs(q_adj[u].begin(), q_adj[u].end());
        std::sort(nbrs.begin(), nbrs.end(), std::greater<int>());
        for (int v : nbrs) if (!seen[v]) stack.push_back(v);
    }
    if (nodes.empty()) nodes.push_back(0);
    return nodes;
}

std::vector<int> order_with_neugn(
    const DemoArgs& args,
    const Query& q,
    int depth,
    int qnode,
    const std::vector<int>& local_cands,
    const std::unordered_map<int, int>& mapping,
    const std::unordered_map<int, std::vector<int>>& path_map,
    const TokenizerMeta& tok
) {
    if (depth >= args.nav_depth || local_cands.empty()) return local_cands;

    auto p_it = path_map.find(q.query_id);
    std::vector<int> path_nodes;
    if (p_it != path_map.end()) path_nodes = p_it->second;
    else path_nodes = build_path_fallback(q, build_adj(q.n, q.edge_src, q.edge_dst));

    std::unordered_map<int, int> node2sub;
    for (size_t i = 0; i < path_nodes.size(); ++i) node2sub[path_nodes[i]] = static_cast<int>(i % tok.sub_node_id_size);

    std::vector<int64_t> tokens;
    std::vector<int64_t> subnodes;
    tokens.reserve(path_nodes.size() + 1);
    subnodes.reserve(path_nodes.size() + 1);
    tokens.push_back(tok.sos_id);
    subnodes.push_back(node2sub.count(qnode) ? node2sub[qnode] : 0);
    for (int pn : path_nodes) {
        if (pn == qnode) tokens.push_back(tok.sos_id);
        else if (mapping.count(pn)) tokens.push_back(mapping.at(pn));
        else tokens.push_back(tok.padding_id);
        subnodes.push_back(node2sub[pn]);
    }
    std::vector<int64_t> token_len = {static_cast<int64_t>(tokens.size())};

    std::vector<int64_t> q_edge_i64;
    q_edge_i64.reserve(2 * q.edge_src.size());
    for (int v : q.edge_src) q_edge_i64.push_back(v);
    for (int v : q.edge_dst) q_edge_i64.push_back(v);
    std::vector<int64_t> q_labels_i64(q.labels.begin(), q.labels.end());

    const std::string input_dir = args.export_dir + "/input";
    write_i64_bin(input_dir + "/graph_edge_index.bin", q_edge_i64);
    write_i64_bin(input_dir + "/graph_feat_id.bin", q_labels_i64);
    write_i64_bin(input_dir + "/tokens.bin", tokens);
    write_i64_bin(input_dir + "/subnode_ids.bin", subnodes);
    write_i64_bin(input_dir + "/token_mask_len.bin", token_len);

    NeuGNCudaModel model;
    model.load(args.export_dir);
    model.forward_full_model();
    std::string tmp_out = args.export_dir + "/demo_input/.tmp_demo_logits.bin";
    model.save_output(tmp_out);
    auto logits = read_binary_float32(tmp_out);
    if (logits.empty()) throw std::runtime_error("Empty logits from CUDA model");

    std::unordered_map<int, float> score;
    for (int c : local_cands) {
        if (c < 0 || c >= static_cast<int>(logits.size())) {
            throw std::runtime_error("Candidate id out of logits range: " + std::to_string(c));
        }
        score[c] = logits[c];
    }
    std::vector<int> out = local_cands;
    std::sort(out.begin(), out.end(), [&](int a, int b) {
        if (score[a] != score[b]) return score[a] > score[b];
        return a < b;
    });
    return out;
}

MatchResult enumerate_first(
    const Query& q,
    const std::vector<int>& q_order,
    const std::vector<std::unordered_set<int>>& q_adj,
    const std::vector<int>& data_labels,
    const std::vector<std::unordered_set<int>>& d_adj,
    std::function<std::vector<int>(int, int, const std::vector<int>&, const std::unordered_map<int, int>&)> order_fn,
    const std::optional<int>& max_steps
) {
    auto t0 = std::chrono::high_resolution_clock::now();
    MatchResult r;
    std::unordered_map<int, int> mapping;
    std::unordered_set<int> used_data;

    std::function<bool(int)> dfs = [&](int depth) {
        if (max_steps.has_value() && r.fms >= *max_steps) return false;
        if (depth == static_cast<int>(q_order.size())) {
            r.found = true;
            r.first_match = mapping;
            return true;
        }
        int qn = q_order[depth];
        auto lc = local_candidates(qn, q, data_labels, q_adj, d_adj);
        auto ordered = order_fn(depth, qn, lc, mapping);
        for (int dn : ordered) {
            r.fms += 1;
            if (max_steps.has_value() && r.fms > *max_steps) return false;
            if (!consistent(qn, dn, mapping, used_data, q_adj, d_adj)) continue;
            mapping[qn] = dn;
            used_data.insert(dn);
            if (dfs(depth + 1)) return true;
            used_data.erase(dn);
            mapping.erase(qn);
        }
        return false;
    };
    dfs(0);
    auto t1 = std::chrono::high_resolution_clock::now();
    r.elapsed = std::chrono::duration<double>(t1 - t0).count();
    return r;
}

void write_csv(const std::string& path, const std::vector<Row>& rows) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to open csv output: " + path);
    out << "query_id,query_size,query_edges,query_avg_degree,density_group,baseline_fms,neugn_fms,improvement_percent,baseline_time,neugn_time,baseline_found,neugn_found,baseline_mps,neugn_mps\n";
    for (const auto& r : rows) {
        out << r.query_id << "," << r.query_size << "," << r.query_edges << "," << r.query_avg_degree << "," << r.density_group
            << "," << r.baseline_fms << "," << r.neugn_fms << ",";
        if (r.improvement_percent.has_value()) out << *r.improvement_percent;
        out << "," << r.baseline_time << "," << r.neugn_time << ","
            << (r.baseline_found ? "true" : "false") << "," << (r.neugn_found ? "true" : "false") << ",";
        if (r.baseline_mps.has_value()) out << *r.baseline_mps;
        out << ",";
        if (r.neugn_mps.has_value()) out << *r.neugn_mps;
        out << "\n";
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        DemoArgs args = parse_args(argc, argv);
        CHECK_CUDA(cudaSetDevice(args.device_id));

        auto cfg = parse_config_txt(args.export_dir + "/config.txt");
        (void)cfg;
        auto manifest = parse_manifest_tsv(args.export_dir + "/manifest.tsv");
        (void)manifest;

        std::ifstream nnf(args.export_dir + "/demo_input/data_num_nodes.txt");
        if (!nnf) throw std::runtime_error("Missing data_num_nodes.txt");
        int num_nodes = 0;
        nnf >> num_nodes;
        auto edges_flat = read_i32_bin(args.graph_bin);
        auto labels = read_i32_bin(args.export_dir + "/demo_input/data_labels_i32.bin");
        if (edges_flat.size() % 2 != 0) throw std::runtime_error("Invalid data_edges_i32.bin");
        int e = static_cast<int>(edges_flat.size() / 2);
        std::vector<int> data_src(edges_flat.begin(), edges_flat.begin() + e);
        std::vector<int> data_dst(edges_flat.begin() + e, edges_flat.end());
        auto d_adj = build_adj(num_nodes, data_src, data_dst);

        auto tok = load_tokenizer_meta(args.export_dir + "/demo_input/tokenizer_meta.txt");
        auto queries = load_queries_bin(args.query_bin);
        auto path_map = load_query_paths_bin(args.export_dir + "/demo_input/query_paths.bin");
        if (queries.size() > static_cast<size_t>(args.num_queries)) queries.resize(args.num_queries);

        std::vector<Row> rows;
        rows.reserve(queries.size());
        for (size_t i = 0; i < queries.size(); ++i) {
            const auto& q = queries[i];
            try {
                auto q_adj = build_adj(q.n, q.edge_src, q.edge_dst);
                auto q_order = query_order(q, q_adj);
                int q_edges = static_cast<int>(std::accumulate(q_adj.begin(), q_adj.end(), 0, [](int a, const auto& s) { return a + static_cast<int>(s.size()); }) / 2);
                double q_avg_degree = (2.0 * q_edges) / std::max(1, q.n);
                std::string group = q_avg_degree >= 3.0 ? "dense" : "sparse";

                auto base_order = [&](int, int, const std::vector<int>& lc, const std::unordered_map<int, int>&) { return lc; };
                auto neug_order = [&](int depth, int qnode, const std::vector<int>& lc, const std::unordered_map<int, int>& mapping) {
                    return order_with_neugn(args, q, depth, qnode, lc, mapping, path_map, tok);
                };

                auto br = enumerate_first(q, q_order, q_adj, labels, d_adj, base_order, args.max_steps);
                auto nr = enumerate_first(q, q_order, q_adj, labels, d_adj, neug_order, args.max_steps);

                Row row;
                row.query_id = q.query_id;
                row.query_size = q.n;
                row.query_edges = q_edges;
                row.query_avg_degree = q_avg_degree;
                row.density_group = group;
                row.baseline_fms = br.fms;
                row.neugn_fms = nr.fms;
                if (br.found && nr.found && br.fms > 0) row.improvement_percent = (br.fms - nr.fms) * 100.0 / br.fms;
                row.baseline_time = br.elapsed;
                row.neugn_time = nr.elapsed;
                row.baseline_found = br.found;
                row.neugn_found = nr.found;
                rows.push_back(row);

                if (static_cast<int>(i) < args.print_first) {
                    std::cout << "[query " << q.query_id << "] baseline_fms=" << br.fms << " neugn_fms=" << nr.fms << "\n";
                }
            } catch (const std::exception& qe) {
                std::cerr << "[query_error] query_id=" << q.query_id << " " << qe.what() << std::endl;
                throw;
            }
        }

        write_csv(args.output, rows);
        std::cout << "[done] wrote " << rows.size() << " rows to " << args.output << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[demo_cu][error] " << e.what() << std::endl;
        return 1;
    }
}
