#pragma once

#include <cuda_runtime.h>

#include <vector>

constexpr int MAX_QUERY_NODES = 32;
constexpr int MAX_QUERY_EDGES = 256;

struct DeviceGraphCSR {
    int num_nodes = 0;
    int num_edges = 0;
    int num_labels = 0;
    int* row_ptr = nullptr;
    int* col_idx = nullptr;
    int* labels = nullptr;
    int* degree = nullptr;
    int* label_ptr = nullptr;
    int* label_nodes = nullptr;
};

struct HostQueryBatch {
    int num_queries = 0;
    int max_query_nodes = MAX_QUERY_NODES;
    int max_query_edges = MAX_QUERY_EDGES;
    std::vector<int> q_num_nodes;
    std::vector<int> q_num_edges;
    std::vector<int> q_labels;
    std::vector<int> q_adj_row_ptr;
    std::vector<int> q_adj_col_idx;
    std::vector<int> q_degree;
    std::vector<int> q_path_ptr;
    std::vector<int> q_path_nodes;
};

struct DeviceQueryBatch {
    int num_queries = 0;
    int max_query_nodes = MAX_QUERY_NODES;
    int max_query_edges = MAX_QUERY_EDGES;

    int* q_num_nodes = nullptr;
    int* q_num_edges = nullptr;
    int* q_labels = nullptr;
    int* q_adj_row_ptr = nullptr;
    int* q_adj_col_idx = nullptr;
    int* q_degree = nullptr;
    int* q_order = nullptr;
    int* q_path_ptr = nullptr;
    int* q_path_nodes = nullptr;
};

DeviceGraphCSR upload_graph_csr(int num_nodes, const std::vector<int>& src, const std::vector<int>& dst, const std::vector<int>& labels);
DeviceQueryBatch upload_query_batch(const HostQueryBatch& host);

void free_graph_csr(DeviceGraphCSR& g);
void free_query_batch(DeviceQueryBatch& q);
