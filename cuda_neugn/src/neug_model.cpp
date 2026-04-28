#include "neug_model.hpp"

#include "kernels.cuh"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
std::vector<int64_t> read_shape_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open shape file: " + path);
    std::string csv;
    std::getline(in, csv);
    return parse_shape_csv(csv);
}

int cfg_int(const std::unordered_map<std::string, std::string>& cfg, const std::string& key) {
    auto it = cfg.find(key);
    if (it == cfg.end()) throw std::runtime_error("Missing config key: " + key);
    return std::stoi(it->second);
}

float cfg_float(const std::unordered_map<std::string, std::string>& cfg, const std::string& key) {
    auto it = cfg.find(key);
    if (it == cfg.end()) throw std::runtime_error("Missing config key: " + key);
    return std::stof(it->second);
}

std::vector<float> load_weight_by_name(
    const std::unordered_map<std::string, TensorInfo>& manifest,
    const std::string& base,
    const std::string& name
) {
    auto it = manifest.find(name);
    if (it == manifest.end()) throw std::runtime_error("Missing required weight in manifest: " + name);
    return read_binary_float32(base + "/" + it->second.relative_path);
}

void upload_to_device(const std::vector<float>& h, float** d) {
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(d), h.size() * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc failed");
    err = cudaMemcpy(*d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error("cudaMemcpy H2D failed");
}

void upload_to_device_i64(const std::vector<int64_t>& h, int64_t** d) {
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(d), h.size() * sizeof(int64_t));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc failed i64");
    err = cudaMemcpy(*d, h.data(), h.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error("cudaMemcpy H2D failed i64");
}

std::vector<int64_t> make_self_looped(const std::vector<int64_t>& edge, int num_nodes) {
    std::vector<int64_t> out = edge;
    out.reserve(edge.size() + num_nodes);
    for (int i = 0; i < num_nodes; ++i) out.push_back(i);
    return out;
}
}  // namespace

void NeuGNCudaModel::require_weight(const std::string& name) const {
    if (manifest_.find(name) == manifest_.end()) {
        throw std::runtime_error("Missing required weight in manifest: " + name);
    }
}

void NeuGNCudaModel::clear_cuda() {
    if (d_src_) cudaFree(d_src_);
    if (d_dst_) cudaFree(d_dst_);
    if (d_feat_id_) cudaFree(d_feat_id_);
    if (d_tokens_) cudaFree(d_tokens_);
    if (d_subnode_) cudaFree(d_subnode_);
    if (d_deg_) cudaFree(d_deg_);
    if (d_h_) cudaFree(d_h_);
    if (d_tmp_) cudaFree(d_tmp_);
    if (d_graph_) cudaFree(d_graph_);
    if (d_masked_h_) cudaFree(d_masked_h_);
    if (d_q_) cudaFree(d_q_);
    if (d_k_) cudaFree(d_k_);
    if (d_v_) cudaFree(d_v_);
    if (d_scores_) cudaFree(d_scores_);
    if (d_ctx_) cudaFree(d_ctx_);
    if (d_ffn1_) cudaFree(d_ffn1_);
    if (d_ffn3_) cudaFree(d_ffn3_);
    if (d_ffn_hidden_) cudaFree(d_ffn_hidden_);
    if (d_logits_) cudaFree(d_logits_);

    d_src_ = d_dst_ = d_feat_id_ = d_tokens_ = d_subnode_ = nullptr;
    d_deg_ = nullptr;
    d_h_ = d_tmp_ = d_graph_ = d_masked_h_ = nullptr;
    d_q_ = d_k_ = d_v_ = d_scores_ = d_ctx_ = nullptr;
    d_ffn1_ = d_ffn3_ = d_ffn_hidden_ = nullptr;
    d_logits_ = nullptr;
}

void NeuGNCudaModel::load(const std::string& export_dir) {
    clear_cuda();
    export_dir_ = export_dir;
    config_ = parse_config_txt(export_dir + "/config.txt");
    manifest_ = parse_manifest_tsv(export_dir + "/manifest.tsv");

    if (config_.at("encoder_name") != "gcn") throw std::runtime_error("Only encoder_name=gcn supported");
    if (config_.at("decoder_type") != "llama") throw std::runtime_error("Only decoder_type=llama supported");

    num_nodes_ = cfg_int(config_, "num_nodes");
    token_len_ = cfg_int(config_, "token_len");
    dim_ = cfg_int(config_, "decoder_dim");
    n_layers_ = cfg_int(config_, "n_layers");
    n_heads_ = cfg_int(config_, "n_heads");
    head_dim_ = dim_ / n_heads_;
    norm_eps_ = cfg_float(config_, "norm_eps");

    auto edge_all = read_binary_int64(export_dir + "/input/graph_edge_index.bin");
    if (edge_all.size() % 2 != 0) throw std::runtime_error("graph_edge_index.bin invalid");
    num_edges_ = static_cast<int>(edge_all.size() / 2);
    edge_src_h_.assign(edge_all.begin(), edge_all.begin() + num_edges_);
    edge_dst_h_.assign(edge_all.begin() + num_edges_, edge_all.end());

    edge_src_h_ = make_self_looped(edge_src_h_, num_nodes_);
    edge_dst_h_ = make_self_looped(edge_dst_h_, num_nodes_);

    feat_id_h_ = read_binary_int64(export_dir + "/input/graph_feat_id.bin");
    tokens_h_ = read_binary_int64(export_dir + "/input/tokens.bin");
    subnode_h_ = read_binary_int64(export_dir + "/input/subnode_ids.bin");
    token_mask_len_h_ = read_binary_int64(export_dir + "/input/token_mask_len.bin");

    if (tokens_h_.size() != static_cast<size_t>(token_len_)) throw std::runtime_error("tokens length mismatch");
    if (subnode_h_.size() != static_cast<size_t>(token_len_)) throw std::runtime_error("subnode length mismatch");

    // minimal critical weights
    require_weight("encoder.value_embedding.weight");
    require_weight("decoder.tok_embeddings.weight");
    require_weight("decoder.node_embeddings.ne");
    require_weight("decoder.type_embeddings.weight");
    require_weight("decoder.pos_embeddings.pe");
    require_weight("decoder.norm.weight");
    for (int i = 0; i < n_layers_; ++i) {
        require_weight("decoder.layers." + std::to_string(i) + ".attention.wq.weight");
        require_weight("decoder.layers." + std::to_string(i) + ".attention.wk.weight");
        require_weight("decoder.layers." + std::to_string(i) + ".attention.wv.weight");
        require_weight("decoder.layers." + std::to_string(i) + ".attention.wo.weight");
        require_weight("decoder.layers." + std::to_string(i) + ".attention_norm.weight");
        require_weight("decoder.layers." + std::to_string(i) + ".ffn_norm.weight");
        require_weight("decoder.layers." + std::to_string(i) + ".feed_forward.w1.weight");
        require_weight("decoder.layers." + std::to_string(i) + ".feed_forward.w2.weight");
        require_weight("decoder.layers." + std::to_string(i) + ".feed_forward.w3.weight");
    }

    output_shape_ = read_shape_file(export_dir + "/python_output.shape");

    // upload static inputs
    upload_to_device_i64(edge_src_h_, &d_src_);
    upload_to_device_i64(edge_dst_h_, &d_dst_);
    upload_to_device_i64(feat_id_h_, &d_feat_id_);
    upload_to_device_i64(tokens_h_, &d_tokens_);
    upload_to_device_i64(subnode_h_, &d_subnode_);

    cudaMalloc(reinterpret_cast<void**>(&d_deg_), num_nodes_ * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&d_h_), num_nodes_ * dim_ * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_tmp_), num_nodes_ * dim_ * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_graph_), dim_ * sizeof(float));

    int seq = 1 + token_len_;
    cudaMalloc(reinterpret_cast<void**>(&d_masked_h_), seq * dim_ * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_q_), seq * dim_ * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_k_), seq * dim_ * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_v_), seq * dim_ * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_ctx_), seq * dim_ * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_scores_), n_heads_ * seq * seq * sizeof(float));

    // ffn dim from first layer w1
    auto w1_shape = manifest_.at("decoder.layers.0.feed_forward.w1.weight").shape;
    int ffn_dim = static_cast<int>(w1_shape[0]);
    cudaMalloc(reinterpret_cast<void**>(&d_ffn1_), seq * ffn_dim * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_ffn3_), seq * ffn_dim * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_ffn_hidden_), seq * ffn_dim * sizeof(float));

    int vocab = cfg_int(config_, "output_dim");
    cudaMalloc(reinterpret_cast<void**>(&d_logits_), vocab * sizeof(float));
}

void NeuGNCudaModel::forward_full_model() {
    int e = static_cast<int>(edge_src_h_.size());
    int valid_rows = static_cast<int>(token_mask_len_h_[0]) + 1;
    int seq = 1 + token_len_;

    // ----- Encoder: GCN -----
    auto val_emb = load_weight_by_name(manifest_, export_dir_, "encoder.value_embedding.weight");
    float* d_val_emb = nullptr;
    upload_to_device(val_emb, &d_val_emb);
    launch_embedding_lookup_kernel(d_feat_id_, d_val_emb, d_h_, num_nodes_, dim_);

    for (int l = 0; l < cfg_int(config_, "encoder_layers"); ++l) {
        std::string prefix = "encoder.convs." + std::to_string(l) + ".linear.";
        auto w = load_weight_by_name(manifest_, export_dir_, prefix + "weight");
        auto b = load_weight_by_name(manifest_, export_dir_, prefix + "bias");
        float *d_w = nullptr, *d_b = nullptr;
        upload_to_device(w, &d_w);
        upload_to_device(b, &d_b);

        launch_zero_int(d_deg_, num_nodes_);
        launch_degree_kernel(d_dst_, d_deg_, e);
        launch_zero_float(d_tmp_, num_nodes_ * dim_);
        launch_gcn_aggregate_kernel(d_src_, d_dst_, d_deg_, d_h_, d_tmp_, e, dim_);
        launch_linear_kernel(d_tmp_, d_w, d_b, d_h_, num_nodes_, dim_, dim_);
        launch_relu_kernel(d_h_, num_nodes_ * dim_);

        cudaFree(d_w);
        cudaFree(d_b);
    }
    cudaFree(d_val_emb);

    launch_max_pool_kernel(d_h_, d_graph_, num_nodes_, dim_);

    // ----- Build decoder input h [seq, dim] -----
    auto tok_emb = load_weight_by_name(manifest_, export_dir_, "decoder.tok_embeddings.weight");
    auto node_emb = load_weight_by_name(manifest_, export_dir_, "decoder.node_embeddings.ne");
    auto type_emb = load_weight_by_name(manifest_, export_dir_, "decoder.type_embeddings.weight");
    auto pos_emb = load_weight_by_name(manifest_, export_dir_, "decoder.pos_embeddings.pe");

    float *d_tok = nullptr, *d_node = nullptr, *d_type = nullptr, *d_pos = nullptr;
    upload_to_device(tok_emb, &d_tok);
    upload_to_device(node_emb, &d_node);
    upload_to_device(type_emb, &d_type);
    upload_to_device(pos_emb, &d_pos);

    // token embeddings into rows [1..]
    launch_embedding_lookup_kernel(d_tokens_, d_tok, d_masked_h_ + dim_, token_len_, dim_);
    launch_embedding_lookup_kernel(d_subnode_, d_node, d_tmp_, token_len_, dim_);
    launch_add_inplace_kernel(d_masked_h_ + dim_, d_tmp_, token_len_ * dim_);
    launch_add_row_vector_inplace(d_masked_h_ + dim_, d_type + 0 * dim_, token_len_, dim_);

    // graph token row 0
    launch_copy_kernel(d_graph_, d_masked_h_, dim_);
    launch_add_inplace_kernel(d_masked_h_, d_type + 1 * dim_, dim_);

    // + position embedding
    launch_add_inplace_kernel(d_masked_h_, d_pos, seq * dim_);

    cudaFree(d_tok); cudaFree(d_node); cudaFree(d_type); cudaFree(d_pos);

    // ----- Transformer layers -----
    auto norm_w = load_weight_by_name(manifest_, export_dir_, "decoder.norm.weight");
    float* d_norm_w = nullptr;
    upload_to_device(norm_w, &d_norm_w);

    for (int l = 0; l < n_layers_; ++l) {
        std::string p = "decoder.layers." + std::to_string(l) + ".";
        auto attn_norm_w = load_weight_by_name(manifest_, export_dir_, p + "attention_norm.weight");
        auto ffn_norm_w = load_weight_by_name(manifest_, export_dir_, p + "ffn_norm.weight");

        auto wq = load_weight_by_name(manifest_, export_dir_, p + "attention.wq.weight");
        auto wk = load_weight_by_name(manifest_, export_dir_, p + "attention.wk.weight");
        auto wv = load_weight_by_name(manifest_, export_dir_, p + "attention.wv.weight");
        auto wo = load_weight_by_name(manifest_, export_dir_, p + "attention.wo.weight");

        auto w1 = load_weight_by_name(manifest_, export_dir_, p + "feed_forward.w1.weight");
        auto w2 = load_weight_by_name(manifest_, export_dir_, p + "feed_forward.w2.weight");
        auto w3 = load_weight_by_name(manifest_, export_dir_, p + "feed_forward.w3.weight");

        float *d_attn_norm_w=nullptr,*d_ffn_norm_w=nullptr,*d_wq=nullptr,*d_wk=nullptr,*d_wv=nullptr,*d_wo=nullptr,*d_w1=nullptr,*d_w2=nullptr,*d_w3=nullptr;
        upload_to_device(attn_norm_w, &d_attn_norm_w);
        upload_to_device(ffn_norm_w, &d_ffn_norm_w);
        upload_to_device(wq, &d_wq); upload_to_device(wk, &d_wk); upload_to_device(wv, &d_wv); upload_to_device(wo, &d_wo);
        upload_to_device(w1, &d_w1); upload_to_device(w2, &d_w2); upload_to_device(w3, &d_w3);

        // attn norm -> tmp
        launch_rmsnorm_kernel(d_masked_h_, d_attn_norm_w, d_tmp_, seq, dim_, norm_eps_);

        launch_linear_kernel(d_tmp_, d_wq, nullptr, d_q_, seq, dim_, dim_);
        launch_linear_kernel(d_tmp_, d_wk, nullptr, d_k_, seq, dim_, dim_);
        launch_linear_kernel(d_tmp_, d_wv, nullptr, d_v_, seq, dim_, dim_);

        launch_attention_scores_kernel(d_q_, d_k_, d_scores_, seq, n_heads_, head_dim_);
        launch_attention_mask_row_kernel(d_scores_, seq, n_heads_, valid_rows);
        launch_softmax_rows_kernel(d_scores_, n_heads_ * seq, seq);
        launch_attention_weighted_sum_kernel(d_scores_, d_v_, d_ctx_, seq, n_heads_, head_dim_);

        launch_linear_kernel(d_ctx_, d_wo, nullptr, d_tmp_, seq, dim_, dim_);
        launch_add_inplace_kernel(d_masked_h_, d_tmp_, seq * dim_);

        // FFN
        launch_rmsnorm_kernel(d_masked_h_, d_ffn_norm_w, d_tmp_, seq, dim_, norm_eps_);

        int ffn_dim = static_cast<int>(manifest_.at(p + "feed_forward.w1.weight").shape[0]);
        launch_linear_kernel(d_tmp_, d_w1, nullptr, d_ffn1_, seq, dim_, ffn_dim);
        launch_linear_kernel(d_tmp_, d_w3, nullptr, d_ffn3_, seq, dim_, ffn_dim);
        launch_copy_kernel(d_ffn1_, d_ffn_hidden_, seq * ffn_dim);
        launch_silu_mul_kernel(d_ffn_hidden_, d_ffn3_, seq * ffn_dim);
        launch_linear_kernel(d_ffn_hidden_, d_w2, nullptr, d_tmp_, seq, ffn_dim, dim_);
        launch_add_inplace_kernel(d_masked_h_, d_tmp_, seq * dim_);

        cudaFree(d_attn_norm_w); cudaFree(d_ffn_norm_w); cudaFree(d_wq); cudaFree(d_wk); cudaFree(d_wv); cudaFree(d_wo); cudaFree(d_w1); cudaFree(d_w2); cudaFree(d_w3);
    }

    // final norm
    launch_rmsnorm_kernel(d_masked_h_, d_norm_w, d_tmp_, seq, dim_, norm_eps_);

    // take row 1
    launch_copy_row_kernel(d_tmp_, d_graph_, 1, dim_);

    // output mlp
    auto ow0 = load_weight_by_name(manifest_, export_dir_, "decoder.output.0.weight");
    auto ob0 = load_weight_by_name(manifest_, export_dir_, "decoder.output.0.bias");
    auto ow2 = load_weight_by_name(manifest_, export_dir_, "decoder.output.2.weight");
    auto ob2 = load_weight_by_name(manifest_, export_dir_, "decoder.output.2.bias");
    float *d_ow0=nullptr,*d_ob0=nullptr,*d_ow2=nullptr,*d_ob2=nullptr;
    upload_to_device(ow0,&d_ow0); upload_to_device(ob0,&d_ob0); upload_to_device(ow2,&d_ow2); upload_to_device(ob2,&d_ob2);

    int hid = static_cast<int>(ob0.size());
    int vocab = static_cast<int>(ob2.size());
    launch_linear_kernel(d_graph_, d_ow0, d_ob0, d_ffn1_, 1, dim_, hid);
    launch_gelu_kernel(d_ffn1_, hid);
    launch_linear_kernel(d_ffn1_, d_ow2, d_ob2, d_logits_, 1, hid, vocab);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) throw std::runtime_error("CUDA forward failed");

    output_host_.resize(vocab);
    err = cudaMemcpy(output_host_.data(), d_logits_, vocab * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error("cudaMemcpy output failed");
    output_shape_ = {1, 1, vocab};

    cudaFree(d_norm_w); cudaFree(d_ow0); cudaFree(d_ob0); cudaFree(d_ow2); cudaFree(d_ob2);
}

void NeuGNCudaModel::save_output(const std::string& path) const {
    if (output_host_.empty()) throw std::runtime_error("No output available. Run forward_full_model first.");
    write_binary_float32(path, output_host_);
}

std::vector<float> NeuGNCudaModel::first_values(int k) const {
    int n = std::min<int>(k, static_cast<int>(output_host_.size()));
    return std::vector<float>(output_host_.begin(), output_host_.begin() + n);
}
