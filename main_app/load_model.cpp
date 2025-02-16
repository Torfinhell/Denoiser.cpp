#include "load_model.h"
#include <torch/script.h> 

void load_tensor(const torch::jit::script::Module& module){
    
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

