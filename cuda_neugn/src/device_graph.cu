#include "device_graph.cuh"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace {
void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
}

void copy_i32(const std::vector<int>& h, int** d) {
    if (h.empty()) {
        *d = nullptr;
        return;
    }
    cuda_check(cudaMalloc(reinterpret_cast<void**>(d), h.size() * sizeof(int)), "cudaMalloc i32");
    cuda_check(cudaMemcpy(*d, h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy i32 H2D");
}
}

DeviceGraphCSR upload_graph_csr(int num_nodes, const std::vector<int>& src, const std::vector<int>& dst, const std::vector<int>& labels) {
    if (src.size() != dst.size()) throw std::runtime_error("src/dst size mismatch");
    if (static_cast<int>(labels.size()) != num_nodes) throw std::runtime_error("labels size mismatch");

    DeviceGraphCSR out;
    out.num_nodes = num_nodes;

    std::vector<std::pair<int, int>> edges;
    edges.reserve(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        int s = src[i], d = dst[i];
        if (s < 0 || s >= num_nodes || d < 0 || d >= num_nodes || s == d) continue;
        edges.emplace_back(s, d);
    }
    std::sort(edges.begin(), edges.end());
    out.num_edges = static_cast<int>(edges.size());

    std::vector<int> row_ptr(num_nodes + 1, 0);
    for (const auto& e : edges) row_ptr[e.first + 1]++;
    std::partial_sum(row_ptr.begin(), row_ptr.end(), row_ptr.begin());

    std::vector<int> col_idx(out.num_edges);
    std::vector<int> cur = row_ptr;
    for (const auto& e : edges) col_idx[cur[e.first]++] = e.second;

    std::vector<int> degree(num_nodes, 0);
    for (int i = 0; i < num_nodes; ++i) degree[i] = row_ptr[i + 1] - row_ptr[i];

    int max_label = 0;
    for (int x : labels) max_label = std::max(max_label, x);
    out.num_labels = max_label + 1;
    std::vector<int> label_ptr(out.num_labels + 1, 0);
    for (int x : labels) if (x >= 0 && x < out.num_labels) label_ptr[x + 1]++;
    std::partial_sum(label_ptr.begin(), label_ptr.end(), label_ptr.begin());
    std::vector<int> label_nodes(num_nodes, 0);
    std::vector<int> lcur = label_ptr;
    for (int i = 0; i < num_nodes; ++i) {
        int x = labels[i];
        if (x >= 0 && x < out.num_labels) label_nodes[lcur[x]++] = i;
    }

    copy_i32(row_ptr, &out.row_ptr);
    copy_i32(col_idx, &out.col_idx);
    copy_i32(labels, &out.labels);
    copy_i32(degree, &out.degree);
    copy_i32(label_ptr, &out.label_ptr);
    copy_i32(label_nodes, &out.label_nodes);
    return out;
}

DeviceQueryBatch upload_query_batch(const HostQueryBatch& host) {
    DeviceQueryBatch out;
    out.num_queries = host.num_queries;
    out.max_query_nodes = host.max_query_nodes;
    out.max_query_edges = host.max_query_edges;

    copy_i32(host.q_num_nodes, &out.q_num_nodes);
    copy_i32(host.q_num_edges, &out.q_num_edges);
    copy_i32(host.q_labels, &out.q_labels);
    copy_i32(host.q_adj_row_ptr, &out.q_adj_row_ptr);
    copy_i32(host.q_adj_col_idx, &out.q_adj_col_idx);
    copy_i32(host.q_degree, &out.q_degree);
    copy_i32(host.q_path_ptr, &out.q_path_ptr);
    copy_i32(host.q_path_nodes, &out.q_path_nodes);

    cuda_check(cudaMalloc(reinterpret_cast<void**>(&out.q_order), host.q_labels.size() * sizeof(int)), "cudaMalloc q_order");
    return out;
}

void free_graph_csr(DeviceGraphCSR& g) {
    if (g.row_ptr) cudaFree(g.row_ptr);
    if (g.col_idx) cudaFree(g.col_idx);
    if (g.labels) cudaFree(g.labels);
    if (g.degree) cudaFree(g.degree);
    if (g.label_ptr) cudaFree(g.label_ptr);
    if (g.label_nodes) cudaFree(g.label_nodes);
    g = {};
}

void free_query_batch(DeviceQueryBatch& q) {
    if (q.q_num_nodes) cudaFree(q.q_num_nodes);
    if (q.q_num_edges) cudaFree(q.q_num_edges);
    if (q.q_labels) cudaFree(q.q_labels);
    if (q.q_adj_row_ptr) cudaFree(q.q_adj_row_ptr);
    if (q.q_adj_col_idx) cudaFree(q.q_adj_col_idx);
    if (q.q_degree) cudaFree(q.q_degree);
    if (q.q_order) cudaFree(q.q_order);
    if (q.q_path_ptr) cudaFree(q.q_path_ptr);
    if (q.q_path_nodes) cudaFree(q.q_path_nodes);
    q = {};
}
