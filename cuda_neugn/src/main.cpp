#include "neug_model.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    std::string export_dir = "./cuda_export/wikics";
    std::string output = "./cuda_export/wikics/cuda_output.bin";
    std::string mode = "full";
    int print_first = 10;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
            return std::string(argv[++i]);
        };

        if (a == "--export_dir") export_dir = next(a);
        else if (a == "--output") output = next(a);
        else if (a == "--mode") mode = next(a);
        else if (a == "--print_first") print_first = std::stoi(next(a));
        else throw std::runtime_error("Unknown argument: " + a);
    }

    if (mode != "full") {
        throw std::runtime_error("Only --mode full is currently supported.");
    }

    NeuGNCudaModel model;
    model.load(export_dir);
    model.forward_full_model();
    model.save_output(output);

    auto vals = model.first_values(print_first);
    std::cout << "[cuda] output shape=";
    for (size_t i = 0; i < model.output_shape().size(); ++i) {
        std::cout << model.output_shape()[i] << (i + 1 == model.output_shape().size() ? "" : ",");
    }
    std::cout << "\n[cuda] first " << vals.size() << " values:";
    for (float v : vals) std::cout << " " << v;
    std::cout << std::endl;
    return 0;
}
