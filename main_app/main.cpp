#include "layers.h"
#include "load_model.h"
#include "denoiser.h"
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
    // print_model_layers(module);

    return 0;
}
