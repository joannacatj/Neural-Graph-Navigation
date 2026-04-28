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

    std::vector<float> python_output_host_;
    std::vector<int64_t> output_shape_;

    float* d_python_output_ = nullptr;
    float* d_output_ = nullptr;
    size_t output_numel_ = 0;

    std::vector<float> output_host_;

    void require_weight(const std::string& name) const;
    void clear_cuda();
};
