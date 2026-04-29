#pragma once

#include "device_graph.cuh"

#include <string>
#include <vector>

struct DeviceMatcherConfig {
    int nav_depth = 10;
    int max_steps = 1000000;
    int max_matches = 1;
    int vocab_size = 0;
    int sos_id = 1;
    int padding_id = 0;
    int sub_node_id_size = 32;
    int token_len = 32;
    int decoder_dim = 0;
    int n_layers = 0;
    int n_heads = 0;
};

struct DeviceNeuGNWeights {
    const float* token_embedding = nullptr;  // [vocab_size, decoder_dim]
    int vocab_size = 0;
    int decoder_dim = 0;
};

struct DeviceNeuGNWeightsOwner {
    DeviceNeuGNWeights device_view;
    std::vector<float*> owned_buffers;
};

DeviceNeuGNWeightsOwner load_device_neugn_weights(const std::string& export_dir, DeviceMatcherConfig& cfg);
void free_device_neugn_weights(DeviceNeuGNWeightsOwner& owner);

__device__ void device_neugn_score_candidates(
    const DeviceNeuGNWeights& weights,
    const DeviceMatcherConfig& cfg,
    const DeviceQueryBatch& queries,
    int query_id,
    const int* mapping,
    int next_query_node,
    const int* local_candidates,
    int local_count,
    float* out_scores,
    int* error_code
);
