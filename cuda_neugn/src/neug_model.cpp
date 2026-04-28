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
}  // namespace

void NeuGNCudaModel::require_weight(const std::string& name) const {
    if (manifest_.find(name) == manifest_.end()) {
        throw std::runtime_error("Missing required weight in manifest: " + name);
    }
}

void NeuGNCudaModel::clear_cuda() {
    if (d_graph_features_) cudaFree(d_graph_features_);
    if (d_w0_) cudaFree(d_w0_);
    if (d_b0_) cudaFree(d_b0_);
    if (d_w2_) cudaFree(d_w2_);
    if (d_b2_) cudaFree(d_b2_);
    if (d_hidden_) cudaFree(d_hidden_);
    if (d_output_) cudaFree(d_output_);
    d_graph_features_ = d_w0_ = d_b0_ = d_w2_ = d_b2_ = d_hidden_ = d_output_ = nullptr;
}

void NeuGNCudaModel::load(const std::string& export_dir) {
    clear_cuda();
    export_dir_ = export_dir;
    config_ = parse_config_txt(export_dir + "/config.txt");
    manifest_ = parse_manifest_tsv(export_dir + "/manifest.tsv");

    if (config_.at("encoder_name") != "gcn") {
        throw std::runtime_error("Only encoder_name=gcn is supported by CUDA inference.");
    }
    if (config_.at("decoder_type") != "llama") {
        throw std::runtime_error("Only decoder_type=llama is supported by CUDA inference.");
    }

    require_weight("decoder.output.0.weight");
    require_weight("decoder.output.0.bias");
    require_weight("decoder.output.2.weight");
    require_weight("decoder.output.2.bias");

    // Current CUDA path computes decoder output head from exported graph features.
    graph_features_shape_ = read_shape_file(export_dir + "/python_graph_features.shape");
    if (graph_features_shape_.size() != 3 || graph_features_shape_[0] != 1 || graph_features_shape_[1] != 1) {
        throw std::runtime_error("python_graph_features.shape must be [1,1,dim]");
    }
    graph_features_host_ = read_binary_float32(export_dir + "/python_graph_features.bin");

    output_shape_ = read_shape_file(export_dir + "/python_output.shape");
    output_numel_ = numel_of_shape(output_shape_);

    w0_host_ = load_weight_by_name(manifest_, export_dir, "decoder.output.0.weight");
    b0_host_ = load_weight_by_name(manifest_, export_dir, "decoder.output.0.bias");
    w2_host_ = load_weight_by_name(manifest_, export_dir, "decoder.output.2.weight");
    b2_host_ = load_weight_by_name(manifest_, export_dir, "decoder.output.2.bias");

    const int in_dim = static_cast<int>(graph_features_shape_[2]);
    const int hidden_dim = static_cast<int>(b0_host_.size());
    const int out_dim = static_cast<int>(b2_host_.size());

    if (w0_host_.size() != static_cast<size_t>(hidden_dim * in_dim)) {
        throw std::runtime_error("decoder.output.0.weight shape mismatch");
    }
    if (w2_host_.size() != static_cast<size_t>(out_dim * hidden_dim)) {
        throw std::runtime_error("decoder.output.2.weight shape mismatch");
    }

    upload_to_device(graph_features_host_, &d_graph_features_);
    upload_to_device(w0_host_, &d_w0_);
    upload_to_device(b0_host_, &d_b0_);
    upload_to_device(w2_host_, &d_w2_);
    upload_to_device(b2_host_, &d_b2_);

    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_hidden_), hidden_dim * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc failed for hidden");
    err = cudaMalloc(reinterpret_cast<void**>(&d_output_), out_dim * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc failed for output");
}

void NeuGNCudaModel::forward_full_model() {
    if (!d_graph_features_ || !d_w0_ || !d_w2_) {
        throw std::runtime_error("Model not loaded.");
    }

    const int in_dim = static_cast<int>(graph_features_shape_[2]);
    const int hidden_dim = static_cast<int>(b0_host_.size());
    const int out_dim = static_cast<int>(b2_host_.size());

    // y1 = GELU(x @ W0^T + b0)
    launch_linear_kernel(d_graph_features_, d_w0_, d_b0_, d_hidden_, 1, in_dim, hidden_dim);
    launch_gelu_kernel(d_hidden_, hidden_dim);

    // y2 = y1 @ W2^T + b2
    launch_linear_kernel(d_hidden_, d_w2_, d_b2_, d_output_, 1, hidden_dim, out_dim);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) throw std::runtime_error("CUDA forward kernel launch failed");

    output_host_.resize(static_cast<size_t>(out_dim));
    err = cudaMemcpy(output_host_.data(), d_output_, out_dim * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error("cudaMemcpy D2H failed for output");

    output_shape_ = {1, 1, out_dim};
}

void NeuGNCudaModel::save_output(const std::string& path) const {
    if (output_host_.empty()) {
        throw std::runtime_error("No output available. Run forward_full_model() first.");
    }
    write_binary_float32(path, output_host_);
}

std::vector<float> NeuGNCudaModel::first_values(int k) const {
    int n = std::min<int>(k, static_cast<int>(output_host_.size()));
    return std::vector<float>(output_host_.begin(), output_host_.begin() + n);
}
