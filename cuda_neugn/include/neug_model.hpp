#pragma once

#include "tensor_io.hpp"

#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <vector>

class NeuGNCudaModel {
public:
    void load(const std::string& export_dir);
    void forward_full_model();
    void save_output(const std::string& path) const;
    std::vector<float> first_values(int k) const;
    const std::vector<int64_t>& output_shape() const { return output_shape_; }

private:
    std::string export_dir_;
    std::unordered_map<std::string, std::string> config_;
    std::unordered_map<std::string, TensorInfo> manifest_;

    // inputs
    std::vector<int64_t> edge_src_h_, edge_dst_h_, feat_id_h_, tokens_h_, subnode_h_, token_mask_len_h_;
    int num_nodes_ = 0, num_edges_ = 0, token_len_ = 0;

    // config dims
    int dim_ = 0, n_layers_ = 0, n_heads_ = 0, kv_heads_ = 0, kv_dim_ = 0, head_dim_ = 0;
    float norm_eps_ = 1e-5f;

    // output
    std::vector<int64_t> output_shape_;
    std::vector<float> output_host_;

    // device buffers
    int64_t *d_src_ = nullptr, *d_dst_ = nullptr, *d_feat_id_ = nullptr, *d_tokens_ = nullptr, *d_subnode_ = nullptr;
    int* d_deg_ = nullptr;
    float *d_h_ = nullptr, *d_tmp_ = nullptr, *d_graph_ = nullptr, *d_masked_h_ = nullptr;
    float *d_q_ = nullptr, *d_k_ = nullptr, *d_v_ = nullptr, *d_scores_ = nullptr, *d_ctx_ = nullptr;
    float *d_ffn1_ = nullptr, *d_ffn3_ = nullptr, *d_ffn_hidden_ = nullptr;
    float *d_logits_ = nullptr;

    void require_weight(const std::string& name) const;
    void clear_cuda();
};
