#include <torch/script.h> // One-stop header.
#include <iostream>
#include <vector>

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

int main() {
    // Load the model
    std::string model_path = "../models/SimpleModel_jit.pth"; // Path to your .pth file
    torch::jit::script::Module module;

    try {
        // Deserialize the ScriptModule from file
        module = torch::jit::load(model_path);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << "\n";
        return -1;
    }

    std::cout << "Model loaded successfully\n";

    // Print the model layers and their weights
    print_model_layers(module);

    return 0;
}
