#pragma once

#include "device_graph.cuh"
#include "device_neugn.cuh"

struct DeviceCandidateBuffer {
    int max_candidates_per_qnode = 0;
    int* cand_ptr = nullptr;
    int* cand_count = nullptr;
    int* cand_nodes = nullptr;
};

struct DeviceMatchResult {
    int baseline_fms = 0;
    int neugn_fms = 0;
    int baseline_found = 0;
    int neugn_found = 0;
    int baseline_truncated = 0;
    int neugn_truncated = 0;
    int baseline_matches = 0;
    int neugn_matches = 0;
    float baseline_time_ms = 0;
    float neugn_time_ms = 0;
    int first_match[MAX_QUERY_NODES] = {0};
    int neugn_calls = 0;
    int max_local_candidates = 0;
    int device_error_code = 0;
};

void alloc_candidate_buffer(int num_queries, int max_query_nodes, int max_candidates_per_qnode, DeviceCandidateBuffer& out);
void free_candidate_buffer(DeviceCandidateBuffer& out);

void launch_build_initial_candidates(
    const DeviceGraphCSR& graph,
    const DeviceQueryBatch& queries,
    const DeviceMatcherConfig& cfg,
    DeviceCandidateBuffer& cands,
    DeviceMatchResult* results
);

void launch_build_query_order(const DeviceGraphCSR& graph, const DeviceQueryBatch& queries);

void launch_gpu_baseline_join(
    const DeviceGraphCSR& graph,
    const DeviceQueryBatch& queries,
    const DeviceMatcherConfig& cfg,
    const DeviceCandidateBuffer& cands,
    DeviceMatchResult* results
);

void launch_gpu_neugn_fused_join(
    const DeviceGraphCSR& graph,
    const DeviceQueryBatch& queries,
    const DeviceMatcherConfig& cfg,
    const DeviceCandidateBuffer& cands,
    const DeviceNeuGNWeights& weights,
    DeviceMatchResult* results
);
