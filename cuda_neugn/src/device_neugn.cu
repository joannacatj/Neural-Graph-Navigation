#include "device_neugn.cuh"

#include "tensor_io.hpp"

#include <fstream>
#include <stdexcept>

namespace {
void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
}

void maybe_load_tokenizer_meta(const std::string& path, DeviceMatcherConfig& cfg) {
    std::ifstream in(path);
    if (!in) return;
    std::string line;
    while (std::getline(in, line)) {
        auto p = line.find('=');
        if (p == std::string::npos) continue;
        std::string k = line.substr(0, p);
        std::string v = line.substr(p + 1);
        if (k == "sos_id") cfg.sos_id = std::stoi(v);
        else if (k == "padding_id") cfg.padding_id = std::stoi(v);
        else if (k == "sub_node_id_size") cfg.sub_node_id_size = std::stoi(v);
    }
}
}

DeviceNeuGNWeightsOwner load_device_neugn_weights(const std::string& export_dir, DeviceMatcherConfig& cfg) {
    DeviceNeuGNWeightsOwner owner;
    auto manifest = parse_manifest_tsv(export_dir + "/manifest.tsv");
    auto cfg_txt = parse_config_txt(export_dir + "/config.txt");
    maybe_load_tokenizer_meta(export_dir + "/demo_input/tokenizer_meta.txt", cfg);

    if (cfg_txt.count("encoder_name") && cfg_txt.at("encoder_name") != "gcn") {
        throw std::runtime_error("Only encoder_name=gcn is supported in fused demo");
    }
    if (cfg_txt.count("decoder_type") && cfg_txt.at("decoder_type") != "llama") {
        throw std::runtime_error("Only decoder_type=llama is supported in fused demo");
    }

    if (cfg_txt.count("decoder_dim")) cfg.decoder_dim = std::stoi(cfg_txt.at("decoder_dim"));
    if (cfg_txt.count("n_layers")) cfg.n_layers = std::stoi(cfg_txt.at("n_layers"));
    if (cfg_txt.count("n_heads")) cfg.n_heads = std::stoi(cfg_txt.at("n_heads"));
    if (cfg_txt.count("token_len")) cfg.token_len = std::stoi(cfg_txt.at("token_len"));
    if (cfg_txt.count("vocab_size")) cfg.vocab_size = std::stoi(cfg_txt.at("vocab_size"));

    // Load all float32 tensors once into GPU memory; fused scorer currently reads
    // decoder token embedding directly but the buffers are all resident.
    for (const auto& kv : manifest) {
        const TensorInfo& t = kv.second;
        if (t.dtype != "float32") continue;
        std::vector<float> h = read_binary_float32(export_dir + "/" + t.relative_path);
        float* d = nullptr;
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d), h.size() * sizeof(float)), "cudaMalloc weight");
        cuda_check(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy weight H2D");
        owner.owned_buffers.push_back(d);
        if (t.name == "decoder.embed_tokens.weight") {
            if (t.shape.size() != 2) throw std::runtime_error("decoder.embed_tokens.weight must be 2D");
            owner.device_view.token_embedding = d;
            owner.device_view.vocab_size = static_cast<int>(t.shape[0]);
            owner.device_view.decoder_dim = static_cast<int>(t.shape[1]);
        }
    }

    if (!owner.device_view.token_embedding) {
        throw std::runtime_error("manifest missing decoder.embed_tokens.weight");
    }

    if (cfg.vocab_size <= 0) cfg.vocab_size = owner.device_view.vocab_size;
    if (cfg.decoder_dim <= 0) cfg.decoder_dim = owner.device_view.decoder_dim;
    return owner;
}

void free_device_neugn_weights(DeviceNeuGNWeightsOwner& owner) {
    for (float* p : owner.owned_buffers) {
        if (p) cudaFree(p);
    }
    owner.owned_buffers.clear();
    owner.device_view = {};
}

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
) {
    constexpr int MAX_LOCAL_CANDIDATES = 4096;
    if (local_count > MAX_LOCAL_CANDIDATES) {
        *error_code = 1001;
        return;
    }

    if (!weights.token_embedding || weights.vocab_size <= 0 || weights.decoder_dim <= 0) {
        *error_code = 1002;
        for (int i = 0; i < local_count; ++i) out_scores[i] = -1e9f;
        return;
    }

    // Construct masked tokens in device memory semantics:
    // tokens[0]=sos, path mapped->data_id, next_query_node->sos, unmatched->padding.
    float context = 0.0f;
    int valid = 0;
    int path_beg = queries.q_path_ptr[query_id];
    int path_end = queries.q_path_ptr[query_id + 1];

    for (int p = path_beg; p < path_end && valid < cfg.token_len; ++p) {
        int pn = queries.q_path_nodes[p];
        int token = cfg.padding_id;
        if (valid == 0) token = cfg.sos_id;
        else if (pn == next_query_node) token = cfg.sos_id;
        else if (mapping[pn] >= 0) token = mapping[pn];

        if (token >= 0 && token < weights.vocab_size) {
            const float* e = weights.token_embedding + static_cast<long long>(token) * weights.decoder_dim;
            context += e[0];
        }
        ++valid;
    }

    for (int i = 0; i < local_count; ++i) {
        int cand = local_candidates[i];
        float s = -1e9f;
        if (cand >= 0 && cand < weights.vocab_size) {
            const float* ec = weights.token_embedding + static_cast<long long>(cand) * weights.decoder_dim;
            s = context + ec[0] * 0.5f + static_cast<float>(cand) * 1e-6f;
        }
        out_scores[i] = s;
    }
}
