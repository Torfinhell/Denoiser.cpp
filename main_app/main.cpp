#pragma once
#include "layers.h"
#include "load_model.h"
#include "denoiser.h"
#include "tensors.h"
using namespace Eigen;

int main() {
    std::string base_path = "/home/torfinhell/Denoiser.cpp/tests/test_data/SimpleModel";
    std::string input_path=base_path+"/input.pth";
    std::string model_path=base_path+"/model.pth";
    std::string prediction_path=base_path+"/prediction.pth";
    torch::jit::script::Module prior_input, model, prior_prediction;
    torch::Tensor input, prediction;
    try {
        model = torch::jit::load("/home/torfinhell/Denoiser.cpp/tests/test_data/SimpleModel/model.pth");
        prior_input = torch::jit::load(input_path);
        input = prior_input.attr("prior").toTensor();
        prior_prediction = torch::jit::load(prediction_path);
        prediction = prior_prediction.attr("prior").toTensor();
        std::cout << "Model loaded successfully\n";
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << "\n";
        return -1;
    }
    print_tensor(input);
    print_model_layers(model);
    print_tensor(prediction);

    return 0;
}
