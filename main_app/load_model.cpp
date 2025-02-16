#include "load_model.h"
#include <cassert>
#include <torch/script.h> 
#include "tensors.h"
#include <cassert>
using namespace Eigen;
Tensor1dXf load_vector(const torch::jit::script::Module& module){
    assert(module.children().size()>=1);
    auto first_child=(*module.named_children().begin()).value.named_parameters();
    Tensor1dXf vector(first_child.size());
    size_t counter=0;
    for (const auto& param : first_child) {
        vector(counter++)=param.value.item<float>();
    }
    return vector;
}
void print_model_layers(const torch::jit::script::Module& module) {
    for (const auto& child : module.named_children()) {
        std::cout << "Layer: " << child.name << ", Type: " << child.value.type()->str() << std::endl;
        for (const auto& param : child.value.named_parameters()) {
            std::cout << "  Param: " << param.name << ", Shape: " << param.value.sizes() << std::endl;
            std::cout << "    Weights: " << param.value << std::endl;
        }
        print_model_layers(child.value);
    }
}
void print_tensor(const torch::Tensor& tensor) {
    if (tensor.dim() == 0) {
        std::cout << tensor.item<float>() << std::endl; 
        return;
    }
    std::cout << "Tensor shape: ";
    for (int64_t i = 0; i < tensor.dim(); ++i) {
        std::cout << tensor.size(i) << " ";
    }
    std::cout << std::endl;
    std::cout << "Tensor elements: ";
    for (int64_t i = 0; i < tensor.numel(); ++i) {
        std::cout << tensor[i].item<float>() << " "; 
    }
    std::cout << std::endl;
}

