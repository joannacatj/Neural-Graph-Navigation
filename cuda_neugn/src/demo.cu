#include "neug_model.hpp"
#include "tensor_io.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<int64_t> read_shape_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open shape file: " + path);
    std::string csv;
    std::getline(in, csv);
    return parse_shape_csv(csv);
}

int argmax_index(const std::vector<float>& x) {
    if (x.empty()) return -1;
    return static_cast<int>(std::max_element(x.begin(), x.end()) - x.begin());
}

std::vector<int> topk_indices(const std::vector<float>& x, int k) {
    k = std::max(0, std::min<int>(k, static_cast<int>(x.size())));
    std::vector<int> idx(x.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return x[a] > x[b]; });
    idx.resize(k);
    return idx;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        std::string export_dir = "./cuda_export/wikics";
        std::string output = "./cuda_export/wikics/cuda_output.bin";
        std::string python_ref = "";
        std::string shape_file = "";
        int warmup = 1;
        int runs = 5;
        int topk = 5;

        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto next = [&](const std::string& name) -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
                return std::string(argv[++i]);
            };
            if (a == "--export_dir") export_dir = next(a);
            else if (a == "--output") output = next(a);
            else if (a == "--python_ref") python_ref = next(a);
            else if (a == "--shape") shape_file = next(a);
            else if (a == "--warmup") warmup = std::stoi(next(a));
            else if (a == "--runs") runs = std::stoi(next(a));
            else if (a == "--topk") topk = std::stoi(next(a));
            else throw std::runtime_error("Unknown argument: " + a);
        }

        if (python_ref.empty()) python_ref = export_dir + "/python_output.bin";
        if (shape_file.empty()) shape_file = export_dir + "/python_output.shape";

        NeuGNCudaModel model;
        model.load(export_dir);

        for (int i = 0; i < warmup; ++i) model.forward_full_model();

        std::vector<double> latency_ms;
        latency_ms.reserve(std::max(1, runs));
        for (int i = 0; i < runs; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            model.forward_full_model();
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            latency_ms.push_back(ms);
        }
        model.save_output(output);

        double mean_ms = 0.0;
        if (!latency_ms.empty()) {
            mean_ms = std::accumulate(latency_ms.begin(), latency_ms.end(), 0.0) / static_cast<double>(latency_ms.size());
        }

        std::cout << "[demo] runs=" << runs << " warmup=" << warmup << " mean_latency_ms=" << mean_ms << "\n";

        auto shape = read_shape_file(shape_file);
        size_t expected = numel_of_shape(shape);
        auto a = read_binary_float32(python_ref);
        auto b = read_binary_float32(output);
        if (a.size() != expected || b.size() != expected) {
            throw std::runtime_error("Shape/binary size mismatch in demo compare");
        }

        double max_abs = 0.0;
        double max_rel = 0.0;
        for (size_t i = 0; i < expected; ++i) {
            double abs_e = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
            double denom = std::max(std::abs(static_cast<double>(a[i])), 1e-12);
            double rel_e = abs_e / denom;
            if (abs_e > max_abs) max_abs = abs_e;
            if (rel_e > max_rel) max_rel = rel_e;
        }

        int argmax_a = argmax_index(a);
        int argmax_b = argmax_index(b);
        auto top_a = topk_indices(a, topk);
        auto top_b = topk_indices(b, topk);

        std::cout << "[demo] shape=";
        for (size_t i = 0; i < shape.size(); ++i) std::cout << shape[i] << (i + 1 == shape.size() ? "" : ",");
        std::cout << "\n[demo] max_abs_error=" << max_abs;
        std::cout << "\n[demo] max_rel_error=" << max_rel;
        std::cout << "\n[demo] argmax_python=" << argmax_a << " argmax_cuda=" << argmax_b;
        std::cout << "\n[demo] top" << topk << "_python:";
        for (int i : top_a) std::cout << " (" << i << "," << a[i] << ")";
        std::cout << "\n[demo] top" << topk << "_cuda:";
        for (int i : top_b) std::cout << " (" << i << "," << b[i] << ")";
        std::cout << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[demo][error] " << e.what() << std::endl;
        return 1;
    }
}
