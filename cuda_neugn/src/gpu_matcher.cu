#include "gpu_matcher.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace {
constexpr int MAX_LOCAL_CANDIDATES = 256;

void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
}

__device__ bool has_edge(const DeviceGraphCSR& g, int u, int v) {
    int l = g.row_ptr[u], r = g.row_ptr[u + 1];
    while (l < r) {
        int m = (l + r) >> 1;
        int x = g.col_idx[m];
        if (x == v) return true;
        if (x < v) l = m + 1;
        else r = m;
    }
    return false;
}

__device__ bool consistent(
    const DeviceGraphCSR& g,
    const DeviceQueryBatch& q,
    int qid,
    int qnode,
    int dnode,
    const int* mapping,
    const unsigned char* used
) {
    if (used[dnode]) return false;
    int row_base = qid * (q.max_query_nodes + 1);
    int col_base = qid * q.max_query_edges;
    int beg = q.q_adj_row_ptr[row_base + qnode];
    int end = q.q_adj_row_ptr[row_base + qnode + 1];
    for (int i = beg; i < end; ++i) {
        int nb = q.q_adj_col_idx[col_base + i];
        int mapped = mapping[nb];
        if (mapped >= 0 && !has_edge(g, dnode, mapped)) return false;
    }
    return true;
}

__device__ void sort_by_score(int* cands, float* scores, int n) {
    for (int i = 1; i < n; ++i) {
        int kc = cands[i];
        float ks = scores[i];
        int j = i - 1;
        while (j >= 0 && (scores[j] < ks || (scores[j] == ks && cands[j] > kc))) {
            cands[j + 1] = cands[j];
            scores[j + 1] = scores[j];
            --j;
        }
        cands[j + 1] = kc;
        scores[j + 1] = ks;
    }
}

__global__ void build_initial_candidates_kernel(
    DeviceGraphCSR g,
    DeviceQueryBatch q,
    DeviceMatcherConfig,
    DeviceCandidateBuffer c,
    DeviceMatchResult* results
) {
    int qid = blockIdx.x;
    if (qid >= q.num_queries || threadIdx.x != 0) return;

    int qn = q.q_num_nodes[qid];
    int qbase = qid * q.max_query_nodes;

    for (int u = 0; u < qn; ++u) {
        int label = q.q_labels[qbase + u];
        int qdeg = q.q_degree[qbase + u];
        int out_idx = qbase + u;
        int out_base = out_idx * c.max_candidates_per_qnode;
        int cnt = 0;

        if (label >= 0 && label < g.num_labels) {
            int lb = g.label_ptr[label];
            int le = g.label_ptr[label + 1];
            for (int k = lb; k < le; ++k) {
                int dnode = g.label_nodes[k];
                if (g.degree[dnode] < qdeg) continue;
                if (cnt >= c.max_candidates_per_qnode) {
                    results[qid].device_error_code = 2001;
                    break;
                }
                c.cand_nodes[out_base + cnt] = dnode;
                ++cnt;
            }
        }
        c.cand_count[out_idx] = cnt;
        c.cand_ptr[out_idx] = out_base;
    }
    c.cand_ptr[qbase + qn] = (qbase + qn) * c.max_candidates_per_qnode;
}

__global__ void build_query_order_kernel(DeviceGraphCSR g, DeviceQueryBatch q) {
    int qid = blockIdx.x;
    if (qid >= q.num_queries || threadIdx.x != 0) return;

    int qn = q.q_num_nodes[qid];
    int base = qid * q.max_query_nodes;
    int order[MAX_QUERY_NODES];
    for (int i = 0; i < qn; ++i) order[i] = i;

    for (int i = 1; i < qn; ++i) {
        int key = order[i];
        int key_deg = q.q_degree[base + key];
        int key_lbl = q.q_labels[base + key];
        int key_freq = (key_lbl >= 0 && key_lbl < g.num_labels) ? (g.label_ptr[key_lbl + 1] - g.label_ptr[key_lbl]) : 0;

        int j = i - 1;
        while (j >= 0) {
            int cur = order[j];
            int cur_deg = q.q_degree[base + cur];
            int cur_lbl = q.q_labels[base + cur];
            int cur_freq = (cur_lbl >= 0 && cur_lbl < g.num_labels) ? (g.label_ptr[cur_lbl + 1] - g.label_ptr[cur_lbl]) : 0;
            bool move = (cur_deg < key_deg) || (cur_deg == key_deg && cur_freq > key_freq) ||
                        (cur_deg == key_deg && cur_freq == key_freq && cur > key);
            if (!move) break;
            order[j + 1] = order[j];
            --j;
        }
        order[j + 1] = key;
    }
    for (int i = 0; i < qn; ++i) q.q_order[base + i] = order[i];
}

__device__ int dfs_join(
    const DeviceGraphCSR& g,
    const DeviceQueryBatch& q,
    const DeviceMatcherConfig& cfg,
    const DeviceCandidateBuffer& c,
    int qid,
    bool use_neugn,
    const DeviceNeuGNWeights& weights,
    DeviceMatchResult* out
) {
    int qn = q.q_num_nodes[qid];
    int mapping[MAX_QUERY_NODES];
    int cand_pos[MAX_QUERY_NODES];
    int local_cnt[MAX_QUERY_NODES];
    int local_cands[MAX_QUERY_NODES][MAX_LOCAL_CANDIDATES];
    float scores[MAX_LOCAL_CANDIDATES];

    for (int i = 0; i < qn; ++i) {
        mapping[i] = -1;
        cand_pos[i] = 0;
        local_cnt[i] = -1;
    }

    __shared__ unsigned char used[4096];
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) used[i] = 0;
    __syncthreads();

    int depth = 0;
    int fms = 0;
    int found = 0;
    int truncated = 0;

    while (depth >= 0) {
        if (cfg.max_steps > 0 && fms >= cfg.max_steps) {
            truncated = 1;
            break;
        }
        if (depth == qn) {
            found = 1;
            break;
        }

        int qnode = q.q_order[qid * q.max_query_nodes + depth];
        if (local_cnt[depth] < 0) {
            int idx = qid * q.max_query_nodes + qnode;
            int cnt = c.cand_count[idx];
            if (cnt > MAX_LOCAL_CANDIDATES) {
                cnt = MAX_LOCAL_CANDIDATES;
                out->device_error_code = 2002;
            }
            out->max_local_candidates = (out->max_local_candidates > cnt) ? out->max_local_candidates : cnt;
            int base = idx * c.max_candidates_per_qnode;
            for (int i = 0; i < cnt; ++i) local_cands[depth][i] = c.cand_nodes[base + i];
            local_cnt[depth] = cnt;

            if (use_neugn && depth < cfg.nav_depth) {
                device_neugn_score_candidates(weights, cfg, q, qid, mapping, qnode, local_cands[depth], cnt, scores, &out->device_error_code);
                sort_by_score(local_cands[depth], scores, cnt);
                out->neugn_calls++;
            }
        }

        bool advanced = false;
        for (; cand_pos[depth] < local_cnt[depth]; ++cand_pos[depth]) {
            int dn = local_cands[depth][cand_pos[depth]];
            ++fms;
            if (dn >= 0 && dn < g.num_nodes && dn < 4096 && consistent(g, q, qid, qnode, dn, mapping, used)) {
                mapping[qnode] = dn;
                used[dn] = 1;
                cand_pos[depth]++;
                depth++;
                if (depth < qn) {
                    cand_pos[depth] = 0;
                    local_cnt[depth] = -1;
                }
                advanced = true;
                break;
            }
        }

        if (!advanced) {
            local_cnt[depth] = -1;
            cand_pos[depth] = 0;
            depth--;
            if (depth >= 0) {
                int prev_qnode = q.q_order[qid * q.max_query_nodes + depth];
                int dn = mapping[prev_qnode];
                if (dn >= 0 && dn < 4096) used[dn] = 0;
                mapping[prev_qnode] = -1;
            }
        }
    }

    if (use_neugn) {
        out->neugn_found = found;
        out->neugn_truncated = truncated;
    } else {
        out->baseline_found = found;
        out->baseline_truncated = truncated;
    }
    return fms;
}

__global__ void gpu_baseline_join_kernel(DeviceGraphCSR g, DeviceQueryBatch q, DeviceMatcherConfig cfg, DeviceCandidateBuffer c, DeviceMatchResult* out) {
    int qid = blockIdx.x;
    if (qid >= q.num_queries || threadIdx.x != 0) return;
    out[qid].baseline_fms = dfs_join(g, q, cfg, c, qid, false, DeviceNeuGNWeights{}, &out[qid]);
}

__global__ void gpu_neugn_fused_join_kernel(
    DeviceGraphCSR g,
    DeviceQueryBatch q,
    DeviceMatcherConfig cfg,
    DeviceCandidateBuffer c,
    DeviceNeuGNWeights weights,
    DeviceMatchResult* out
) {
    int qid = blockIdx.x;
    if (qid >= q.num_queries || threadIdx.x != 0) return;
    out[qid].neugn_fms = dfs_join(g, q, cfg, c, qid, true, weights, &out[qid]);
}
}

void alloc_candidate_buffer(int num_queries, int max_query_nodes, int max_candidates_per_qnode, DeviceCandidateBuffer& out) {
    out.max_candidates_per_qnode = max_candidates_per_qnode;
    int slots = num_queries * max_query_nodes;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&out.cand_count), slots * sizeof(int)), "cudaMalloc cand_count");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&out.cand_ptr), (slots + 1) * sizeof(int)), "cudaMalloc cand_ptr");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&out.cand_nodes), static_cast<size_t>(slots) * max_candidates_per_qnode * sizeof(int)), "cudaMalloc cand_nodes");
}

void free_candidate_buffer(DeviceCandidateBuffer& out) {
    if (out.cand_count) cudaFree(out.cand_count);
    if (out.cand_ptr) cudaFree(out.cand_ptr);
    if (out.cand_nodes) cudaFree(out.cand_nodes);
    out = {};
}

void launch_build_initial_candidates(
    const DeviceGraphCSR& graph,
    const DeviceQueryBatch& queries,
    const DeviceMatcherConfig& cfg,
    DeviceCandidateBuffer& cands,
    DeviceMatchResult* results
) {
    build_initial_candidates_kernel<<<queries.num_queries, 32>>>(graph, queries, cfg, cands, results);
}

void launch_build_query_order(const DeviceGraphCSR& graph, const DeviceQueryBatch& queries) {
    build_query_order_kernel<<<queries.num_queries, 1>>>(graph, queries);
}

void launch_gpu_baseline_join(
    const DeviceGraphCSR& graph,
    const DeviceQueryBatch& queries,
    const DeviceMatcherConfig& cfg,
    const DeviceCandidateBuffer& cands,
    DeviceMatchResult* results
) {
    gpu_baseline_join_kernel<<<queries.num_queries, 32>>>(graph, queries, cfg, cands, results);
}

void launch_gpu_neugn_fused_join(
    const DeviceGraphCSR& graph,
    const DeviceQueryBatch& queries,
    const DeviceMatcherConfig& cfg,
    const DeviceCandidateBuffer& cands,
    const DeviceNeuGNWeights& weights,
    DeviceMatchResult* results
) {
    gpu_neugn_fused_join_kernel<<<queries.num_queries, 32>>>(graph, queries, cfg, cands, weights, results);
}
