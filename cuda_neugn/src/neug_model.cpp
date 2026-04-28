#include "neug_model.hpp"

#include "kernels.cuh"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
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
    const size_t nbytes = h.size() * sizeof(float);
    if (nbytes == 0) {
        *d = nullptr;
        return;
    }
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(d), nbytes);
    if (err != cudaSuccess) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        throw std::runtime_error(
            "cudaMalloc failed for float upload (" + std::to_string(nbytes) + " bytes): " +
            std::string(cudaGetErrorString(err)) +
            ", free=" + std::to_string(free_b) + ", total=" + std::to_string(total_b)
        );
    }
    err = cudaMemcpy(*d, h.data(), nbytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy H2D failed for float upload: " + std::string(cudaGetErrorString(err)));
    }
}

void upload_to_device_i64(const std::vector<int64_t>& h, int64_t** d) {
    const size_t nbytes = h.size() * sizeof(int64_t);
    if (nbytes == 0) {
        *d = nullptr;
        return;
    }
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(d), nbytes);
    if (err != cudaSuccess) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        throw std::runtime_error(
            "cudaMalloc failed for int64 upload (" + std::to_string(nbytes) + " bytes): " +
            std::string(cudaGetErrorString(err)) +
            ", free=" + std::to_string(free_b) + ", total=" + std::to_string(total_b)
        );
    }
    err = cudaMemcpy(*d, h.data(), nbytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy H2D failed for int64 upload: " + std::string(cudaGetErrorString(err)));
    }
}

std::vector<int64_t> make_self_looped(const std::vector<int64_t>& edge, int num_nodes) {
    std::vector<int64_t> out = edge;
    out.reserve(edge.size() + num_nodes);
    for (int i = 0; i < num_nodes; ++i) out.push_back(i);
    return out;
}

size_t checked_count_bytes(size_t count, size_t elem_size, const std::string& name) {
    if (count == 0) return 0;
    if (count > std::numeric_limits<size_t>::max() / elem_size) {
        throw std::runtime_error("Allocation size overflow for " + name);
    }
    return count * elem_size;
}

void checked_cuda_malloc(void** ptr, size_t nbytes, const std::string& name) {
    if (nbytes == 0) {
        *ptr = nullptr;
        return;
    }
    cudaError_t err = cudaMalloc(ptr, nbytes);
    if (err != cudaSuccess) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        throw std::runtime_error(
            "cudaMalloc failed for " + name + " (" + std::to_string(nbytes) + " bytes): " +
            std::string(cudaGetErrorString(err)) +
            ", free=" + std::to_string(free_b) + ", total=" + std::to_string(total_b)
        );
    }
}

void check_index_bounds(const std::vector<int64_t>& ids, int64_t limit, const std::string& name) {
    if (limit <= 0) {
        throw std::runtime_error("Invalid bound for " + name + ": " + std::to_string(limit));
    }
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0 || ids[i] >= limit) {
            throw std::runtime_error(
                name + " out of range at index " + std::to_string(i) +
                ": value=" + std::to_string(ids[i]) +
                ", expected in [0, " + std::to_string(limit - 1) + "]"
            );
        }
    }
}

void check_last_cuda_error(const std::string& where) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(where + ": " + std::string(cudaGetErrorString(err)));
    }
}

std::string shape_to_string(const std::vector<int64_t>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        s += std::to_string(shape[i]);
        if (i + 1 != shape.size()) s += ", ";
    }
    s += "]";
    return s;
}

void require_2d_shape(
    const std::unordered_map<std::string, TensorInfo>& manifest,
    const std::string& name,
    int64_t d0,
    int64_t d1
) {
    auto it = manifest.find(name);
    if (it == manifest.end()) throw std::runtime_error("Missing required weight in manifest: " + name);
    const auto& s = it->second.shape;
    if (s.size() != 2 || s[0] != d0 || s[1] != d1) {
        throw std::runtime_error(
            "Unsupported weight shape for " + name +
            ", expected [" + std::to_string(d0) + ", " + std::to_string(d1) + "] got " + shape_to_string(s)
        );
    }
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
    cudaError_t init_err = cudaFree(0);
    if (init_err != cudaSuccess) {
        throw std::runtime_error("CUDA runtime initialization failed: " + std::string(cudaGetErrorString(init_err)));
    }
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
    if (n_heads_ <= 0 || dim_ % n_heads_ != 0) {
        throw std::runtime_error(
            "Invalid attention dims: decoder_dim=" + std::to_string(dim_) +
            ", n_heads=" + std::to_string(n_heads_)
        );
    }
    head_dim_ = dim_ / n_heads_;
    kv_heads_ = n_heads_;
    kv_dim_ = dim_;
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

    // Validate embedding dims.
    enc_in_dim_ = static_cast<int>(manifest_.at("encoder.value_embedding.weight").shape.at(1));
    if (enc_in_dim_ <= 0) {
        throw std::runtime_error("Invalid encoder embedding dim: " + std::to_string(enc_in_dim_));
    }
    require_2d_shape(manifest_, "encoder.value_embedding.weight", manifest_.at("encoder.value_embedding.weight").shape.at(0), enc_in_dim_);
    require_2d_shape(manifest_, "decoder.tok_embeddings.weight", manifest_.at("decoder.tok_embeddings.weight").shape.at(0), dim_);
    require_2d_shape(manifest_, "decoder.node_embeddings.ne", manifest_.at("decoder.node_embeddings.ne").shape.at(0), dim_);
    require_2d_shape(manifest_, "decoder.type_embeddings.weight", manifest_.at("decoder.type_embeddings.weight").shape.at(0), dim_);
    auto pos_it = manifest_.find("decoder.pos_embeddings.pe");
    if (pos_it == manifest_.end()) throw std::runtime_error("Missing required weight in manifest: decoder.pos_embeddings.pe");
    const auto& pos_shape = pos_it->second.shape;
    int64_t pos_rows = -1;
    int64_t pos_dim = -1;
    if (pos_shape.size() == 2) {
        pos_rows = pos_shape[0];
        pos_dim = pos_shape[1];
    } else if (pos_shape.size() == 3 && pos_shape[0] == 1) {
        // PyTorch positional buffer is typically [1, max_len, dim].
        pos_rows = pos_shape[1];
        pos_dim = pos_shape[2];
    } else {
        throw std::runtime_error(
            "Unsupported shape for decoder.pos_embeddings.pe, expected [max_len, dim] or [1, max_len, dim], got " +
            shape_to_string(pos_shape)
        );
    }
    if (pos_dim != dim_) {
        throw std::runtime_error(
            "Unsupported positional embedding dim for decoder.pos_embeddings.pe, expected dim=" +
            std::to_string(dim_) + " got " + std::to_string(pos_dim)
        );
    }
    if (pos_rows < 1 + token_len_) {
        throw std::runtime_error(
            "Positional embedding too short: need at least " + std::to_string(1 + token_len_) +
            " rows, got " + std::to_string(pos_rows)
        );
    }

    // Infer kv projection width from the first layer and enforce consistency across layers.
    {
        const auto& wk0_shape = manifest_.at("decoder.layers.0.attention.wk.weight").shape;
        if (wk0_shape.size() != 2 || wk0_shape[1] != dim_) {
            throw std::runtime_error(
                "Unsupported wk shape at layer 0, expected [kv_dim, " + std::to_string(dim_) +
                "] got " + shape_to_string(wk0_shape)
            );
        }
        kv_dim_ = static_cast<int>(wk0_shape[0]);
        if (kv_dim_ <= 0 || kv_dim_ % head_dim_ != 0) {
            throw std::runtime_error(
                "Unsupported kv_dim=" + std::to_string(kv_dim_) +
                ", must be positive and divisible by head_dim=" + std::to_string(head_dim_)
            );
        }
        kv_heads_ = kv_dim_ / head_dim_;
        if (n_heads_ % kv_heads_ != 0) {
            throw std::runtime_error(
                "Unsupported grouped attention: n_heads=" + std::to_string(n_heads_) +
                " is not divisible by kv_heads=" + std::to_string(kv_heads_)
            );
        }
    }

    for (int i = 0; i < n_layers_; ++i) {
        const std::string p = "decoder.layers." + std::to_string(i) + ".";
        require_2d_shape(manifest_, p + "attention.wq.weight", dim_, dim_);
        require_2d_shape(manifest_, p + "attention.wk.weight", kv_dim_, dim_);
        require_2d_shape(manifest_, p + "attention.wv.weight", kv_dim_, dim_);
        require_2d_shape(manifest_, p + "attention.wo.weight", dim_, dim_);
    }

    // Validate encoder layer widths: first layer maps enc_in_dim -> dim, subsequent layers keep dim -> dim.
    for (int l = 0; l < cfg_int(config_, "encoder_layers"); ++l) {
        std::string p = "encoder.convs." + std::to_string(l) + ".linear.";
        int in_dim = (l == 0) ? enc_in_dim_ : dim_;
        require_2d_shape(manifest_, p + "weight", dim_, in_dim);
        auto b_name = p + "bias";
        auto b_it = manifest_.find(b_name);
        if (b_it == manifest_.end()) throw std::runtime_error("Missing required weight in manifest: " + b_name);
        const auto& b_shape = b_it->second.shape;
        if (b_shape.size() != 1 || b_shape[0] != dim_) {
            throw std::runtime_error(
                "Unsupported bias shape for " + b_name + ", expected [" + std::to_string(dim_) + "] got " + shape_to_string(b_shape)
            );
        }
    }

    // Validate index tensors early to avoid opaque illegal-memory-access errors later in kernels.
    const int64_t value_vocab = manifest_.at("encoder.value_embedding.weight").shape.at(0);
    const int64_t token_vocab = manifest_.at("decoder.tok_embeddings.weight").shape.at(0);
    const int64_t subnode_vocab = manifest_.at("decoder.node_embeddings.ne").shape.at(0);

    check_index_bounds(feat_id_h_, value_vocab, "graph_feat_id");
    check_index_bounds(tokens_h_, token_vocab, "tokens");
    check_index_bounds(subnode_h_, subnode_vocab, "subnode_ids");
    check_index_bounds(edge_src_h_, num_nodes_, "edge_src");
    check_index_bounds(edge_dst_h_, num_nodes_, "edge_dst");

    output_shape_ = read_shape_file(export_dir + "/python_output.shape");

    // upload static inputs
    upload_to_device_i64(edge_src_h_, &d_src_);
    upload_to_device_i64(edge_dst_h_, &d_dst_);
    upload_to_device_i64(feat_id_h_, &d_feat_id_);
    upload_to_device_i64(tokens_h_, &d_tokens_);
    upload_to_device_i64(subnode_h_, &d_subnode_);

    checked_cuda_malloc(reinterpret_cast<void**>(&d_deg_), checked_count_bytes(static_cast<size_t>(num_nodes_), sizeof(int), "d_deg_"), "d_deg_");
    const int encoder_work_dim = std::max(dim_, enc_in_dim_);
    checked_cuda_malloc(reinterpret_cast<void**>(&d_h_), checked_count_bytes(static_cast<size_t>(num_nodes_) * static_cast<size_t>(encoder_work_dim), sizeof(float), "d_h_"), "d_h_");
    checked_cuda_malloc(reinterpret_cast<void**>(&d_tmp_), checked_count_bytes(static_cast<size_t>(num_nodes_) * static_cast<size_t>(encoder_work_dim), sizeof(float), "d_tmp_"), "d_tmp_");
    checked_cuda_malloc(reinterpret_cast<void**>(&d_graph_), checked_count_bytes(static_cast<size_t>(dim_), sizeof(float), "d_graph_"), "d_graph_");

    int seq = 1 + token_len_;
    checked_cuda_malloc(reinterpret_cast<void**>(&d_masked_h_), checked_count_bytes(static_cast<size_t>(seq) * static_cast<size_t>(dim_), sizeof(float), "d_masked_h_"), "d_masked_h_");
    checked_cuda_malloc(reinterpret_cast<void**>(&d_q_), checked_count_bytes(static_cast<size_t>(seq) * static_cast<size_t>(dim_), sizeof(float), "d_q_"), "d_q_");
    checked_cuda_malloc(reinterpret_cast<void**>(&d_k_), checked_count_bytes(static_cast<size_t>(seq) * static_cast<size_t>(kv_dim_), sizeof(float), "d_k_"), "d_k_");
    checked_cuda_malloc(reinterpret_cast<void**>(&d_v_), checked_count_bytes(static_cast<size_t>(seq) * static_cast<size_t>(kv_dim_), sizeof(float), "d_v_"), "d_v_");
    checked_cuda_malloc(reinterpret_cast<void**>(&d_ctx_), checked_count_bytes(static_cast<size_t>(seq) * static_cast<size_t>(dim_), sizeof(float), "d_ctx_"), "d_ctx_");
    checked_cuda_malloc(reinterpret_cast<void**>(&d_scores_), checked_count_bytes(static_cast<size_t>(n_heads_) * static_cast<size_t>(seq) * static_cast<size_t>(seq), sizeof(float), "d_scores_"), "d_scores_");

    // ffn dim from first layer w1
    auto w1_shape = manifest_.at("decoder.layers.0.feed_forward.w1.weight").shape;
    int ffn_dim = static_cast<int>(w1_shape[0]);
    checked_cuda_malloc(reinterpret_cast<void**>(&d_ffn1_), checked_count_bytes(static_cast<size_t>(seq) * static_cast<size_t>(ffn_dim), sizeof(float), "d_ffn1_"), "d_ffn1_");
    checked_cuda_malloc(reinterpret_cast<void**>(&d_ffn3_), checked_count_bytes(static_cast<size_t>(seq) * static_cast<size_t>(ffn_dim), sizeof(float), "d_ffn3_"), "d_ffn3_");
    checked_cuda_malloc(reinterpret_cast<void**>(&d_ffn_hidden_), checked_count_bytes(static_cast<size_t>(seq) * static_cast<size_t>(ffn_dim), sizeof(float), "d_ffn_hidden_"), "d_ffn_hidden_");

    int vocab = cfg_int(config_, "output_dim");
    checked_cuda_malloc(reinterpret_cast<void**>(&d_logits_), checked_count_bytes(static_cast<size_t>(vocab), sizeof(float), "d_logits_"), "d_logits_");
}

void NeuGNCudaModel::forward_full_model() {
    int e = static_cast<int>(edge_src_h_.size());
    int valid_rows = static_cast<int>(token_mask_len_h_[0]) + 1;
    int seq = 1 + token_len_;

    // ----- Encoder: GCN -----
    auto val_emb = load_weight_by_name(manifest_, export_dir_, "encoder.value_embedding.weight");
    float* d_val_emb = nullptr;
    upload_to_device(val_emb, &d_val_emb);
    launch_embedding_lookup_kernel(d_feat_id_, d_val_emb, d_h_, num_nodes_, enc_in_dim_);
    check_last_cuda_error("encoder.value_embedding lookup");

    for (int l = 0; l < cfg_int(config_, "encoder_layers"); ++l) {
        std::string prefix = "encoder.convs." + std::to_string(l) + ".linear.";
        int layer_in_dim = (l == 0) ? enc_in_dim_ : dim_;
        auto w = load_weight_by_name(manifest_, export_dir_, prefix + "weight");
        auto b = load_weight_by_name(manifest_, export_dir_, prefix + "bias");
        float *d_w = nullptr, *d_b = nullptr;
        upload_to_device(w, &d_w);
        upload_to_device(b, &d_b);

        launch_zero_int(d_deg_, num_nodes_);
        launch_degree_kernel(d_dst_, d_deg_, e);
        launch_zero_float(d_tmp_, num_nodes_ * layer_in_dim);
        launch_gcn_aggregate_kernel(d_src_, d_dst_, d_deg_, d_h_, d_tmp_, e, layer_in_dim);
        launch_linear_kernel(d_tmp_, d_w, d_b, d_h_, num_nodes_, layer_in_dim, dim_);
        launch_relu_kernel(d_h_, num_nodes_ * dim_);
        check_last_cuda_error("encoder.gcn layer " + std::to_string(l));

        cudaFree(d_w);
        cudaFree(d_b);
    }
    cudaFree(d_val_emb);

    launch_max_pool_kernel(d_h_, d_graph_, num_nodes_, dim_);
    check_last_cuda_error("encoder.max_pool");

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
    check_last_cuda_error("decoder.input embedding build");

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
        launch_linear_kernel(d_tmp_, d_wk, nullptr, d_k_, seq, dim_, kv_dim_);
        launch_linear_kernel(d_tmp_, d_wv, nullptr, d_v_, seq, dim_, kv_dim_);

        launch_attention_scores_kernel(d_q_, d_k_, d_scores_, seq, n_heads_, kv_heads_, head_dim_);
        launch_attention_mask_row_kernel(d_scores_, seq, n_heads_, valid_rows);
        launch_softmax_rows_kernel(d_scores_, n_heads_ * seq, seq);
        launch_attention_weighted_sum_kernel(d_scores_, d_v_, d_ctx_, seq, n_heads_, kv_heads_, head_dim_);

        launch_linear_kernel(d_ctx_, d_wo, nullptr, d_tmp_, seq, dim_, dim_);
        launch_add_inplace_kernel(d_masked_h_, d_tmp_, seq * dim_);
        check_last_cuda_error("decoder.attention layer " + std::to_string(l));

        // FFN
        launch_rmsnorm_kernel(d_masked_h_, d_ffn_norm_w, d_tmp_, seq, dim_, norm_eps_);

        int ffn_dim = static_cast<int>(manifest_.at(p + "feed_forward.w1.weight").shape[0]);
        launch_linear_kernel(d_tmp_, d_w1, nullptr, d_ffn1_, seq, dim_, ffn_dim);
        launch_linear_kernel(d_tmp_, d_w3, nullptr, d_ffn3_, seq, dim_, ffn_dim);
        launch_copy_kernel(d_ffn1_, d_ffn_hidden_, seq * ffn_dim);
        launch_silu_mul_kernel(d_ffn_hidden_, d_ffn3_, seq * ffn_dim);
        launch_linear_kernel(d_ffn_hidden_, d_w2, nullptr, d_tmp_, seq, ffn_dim, dim_);
        launch_add_inplace_kernel(d_masked_h_, d_tmp_, seq * dim_);
        check_last_cuda_error("decoder.ffn layer " + std::to_string(l));

        cudaFree(d_attn_norm_w); cudaFree(d_ffn_norm_w); cudaFree(d_wq); cudaFree(d_wk); cudaFree(d_wv); cudaFree(d_wo); cudaFree(d_w1); cudaFree(d_w2); cudaFree(d_w3);
    }

    // final norm
    launch_rmsnorm_kernel(d_masked_h_, d_norm_w, d_tmp_, seq, dim_, norm_eps_);
    check_last_cuda_error("decoder.final_norm");

    // take row 1
    launch_copy_row_kernel(d_tmp_, d_graph_, 1, dim_);
    check_last_cuda_error("decoder.select_row");

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
    check_last_cuda_error("decoder.output_mlp");

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) throw std::runtime_error("CUDA forward failed: " + std::string(cudaGetErrorString(err)));

    output_host_.resize(vocab);
    err = cudaMemcpy(output_host_.data(), d_logits_, vocab * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error("cudaMemcpy output failed: " + std::string(cudaGetErrorString(err)));
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
