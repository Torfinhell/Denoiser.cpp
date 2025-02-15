#include "load_model.h"

void print_model_layers(const torch::jit::script::Module& module) {
    // Iterate through the named modules (layers) in the model
    for (const auto& child : module.named_children()) {
        std::cout << "Layer: " << child.name << ", Type: " << child.value.type()->str() << std::endl;

        // If the layer has parameters, print them
        for (const auto& param : child.value.named_parameters()) {
            std::cout << "  Param: " << param.name << ", Shape: " << param.value.sizes() << std::endl;

            // Print the weights (values) of the parameter
            std::cout << "    Weights: " << param.value << std::endl;
        }

        // Recursively print layers of submodules
        print_model_layers(child.value);
    }
}

