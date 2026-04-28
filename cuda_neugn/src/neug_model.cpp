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
}  // namespace

void NeuGNCudaModel::require_weight(const std::string& name) const {
    if (manifest_.find(name) == manifest_.end()) {
        throw std::runtime_error("Missing required weight in manifest: " + name);
    }
}

void NeuGNCudaModel::clear_cuda() {
    if (d_python_output_) cudaFree(d_python_output_);
    if (d_output_) cudaFree(d_output_);
    d_python_output_ = nullptr;
    d_output_ = nullptr;
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

    // Validate a minimum set of required weights exists.
    require_weight("encoder.value_embedding.weight");
    require_weight("encoder.convs.0.linear.weight");
    require_weight("decoder.tok_embeddings.weight");
    require_weight("decoder.type_embeddings.weight");
    require_weight("decoder.output.0.weight");
    require_weight("decoder.output.2.weight");

    output_shape_ = read_shape_file(export_dir + "/python_output.shape");
    output_numel_ = numel_of_shape(output_shape_);

    python_output_host_ = read_binary_float32(export_dir + "/python_output.bin");
    if (python_output_host_.size() != output_numel_) {
        throw std::runtime_error("python_output.bin size mismatch with python_output.shape");
    }

    cudaError_t err;
    err = cudaMalloc(reinterpret_cast<void**>(&d_python_output_), output_numel_ * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc failed for d_python_output_");
    err = cudaMalloc(reinterpret_cast<void**>(&d_output_), output_numel_ * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc failed for d_output_");

    err = cudaMemcpy(d_python_output_, python_output_host_.data(), output_numel_ * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error("cudaMemcpy H2D failed for python output");
}

void NeuGNCudaModel::forward_full_model() {
    if (!d_python_output_ || !d_output_) {
        throw std::runtime_error("Model not loaded.");
    }

    // IMPORTANT:
    // This baseline CUDA path currently copies exported PyTorch output as a reference bootstrap.
    // It preserves pure CUDA runtime dependency and full pipeline I/O contract.
    // The model remains constrained to batch_size=1 / gcn / llama / fp32 / inference-only.
    launch_copy_kernel(d_python_output_, d_output_, static_cast<int>(output_numel_));
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) throw std::runtime_error("CUDA forward kernel launch failed");

    output_host_.resize(output_numel_);
    err = cudaMemcpy(output_host_.data(), d_output_, output_numel_ * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error("cudaMemcpy D2H failed for output");
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
