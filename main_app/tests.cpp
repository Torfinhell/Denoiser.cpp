#include "tests.h"
#include "tensors.h"
#include <ATen/core/TensorBody.h>
#include <chrono>

using namespace Eigen;
Tensor1dXf TorchToEigen(const torch::Tensor& tensor){
    Tensor1dXf eigen_tensor(tensor.dim());
    for(size_t i=0;i<tensor.dim();i++){
        eigen_tensor(i)=tensor[i].item<float>();
    }
    return eigen_tensor;
}
// Tensor2dXf JitToEigen(torch::jit::script::Module tensor){
//     std::vector<int64_t> dimensions = tensor.sizes().vec();
//     for (const auto& child : module.named_children()) {
//         std::cout << "Layer: " << child.name << ", Type: " << child.value.type()->str() << std::endl;
//         for (const auto& param : child.value.named_parameters()) {
//             std::cout << "  Param: " << param.name << ", Shape: " << param.value.sizes() << std::endl;
//             std::cout << "    Weights: " << param.value << std::endl;
//         }
//         print_model_layers(child.value);
//     }
// }
//maybe create converter class for seversl dimensions(if I do it with templates than to long and I dont want to use recursion)
//create time wrapper
void TestSimpleModel(){
    std::string base_path = "/home/torfinhell/Denoiser.cpp/tests/test_data/SimpleModel";
    std::string input_path=base_path+"/input.pth";
    std::string model_path=base_path+"/model.pth";
    std::string prediction_path=base_path+"/prediction.pth";
    torch::jit::script::Module prior_input, model, prior_prediction;
    torch::Tensor input_tensors, prediction_tensors;
    try {
        model = torch::jit::load(model_path);
        prior_input = torch::jit::load(input_path);
        prior_prediction = torch::jit::load(prediction_path);
        input_tensors = prior_input.attr("prior").toTensor();
        prediction_tensors = prior_prediction.attr("prior").toTensor();
        std::cout << "Model loaded successfully\n";
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << "\n";
    }
    print_tensor(input_tensors);
    print_model_layers(model);
    auto start = std::chrono::high_resolution_clock::now();
    print_tensor(prediction_tensors);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Time taken for forward pass: " << duration.count() << " ms" << std::endl;
    // Tensor2dXf weights=JitToEigen(model);
    start = std::chrono::high_resolution_clock::now();
    model.forward({input_tensors}).toTensor();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Time taken for forward pass: " << duration.count() << " ms" << std::endl;

    Eigen::MatrixXd weight(2, 10);
    weight << -0.0697, -0.1289, -0.1705,  0.2072,  0.3073, -0.2763, -0.0724, -0.2576, 0.1368, 0.2347,
               0.1622, -0.0272,  0.3017,  0.0532,  0.2130, -0.0723, -0.0887,  0.0306, 0.1020, -0.1697;

    // Define the bias vector (2)
    Eigen::VectorXd bias(2);
    bias << 6.1365, -30.7464;

    // Define the input matrix (1x10) for a single sample
    Eigen::RowVectorXd input(10);
    input << -0.119387, -0.277944, 0, 0, 0, 0, 0, 0, 0, 0; // Example input
    start = std::chrono::high_resolution_clock::now();
    // Perform the matrix multiplication and add the bias
    Eigen::VectorXd output = (weight * input.transpose()) + bias;
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Time taken for forward pass: " << duration.count() << " ms" << std::endl;
    std::cout << "Output: " << output.transpose() << std::endl;
    // Print the output

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