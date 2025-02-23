#include "tests.h"
#include "layers.h"
#include "tensors.h"
#include <chrono>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include<filesystem>
#include<iostream>
namespace fs = std::filesystem;
using namespace Eigen;
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
void print_1d_tensor(Tensor1dXf tensor) {
    std::cout << "Tensor elements (1D): ";
    for (std::size_t i = 0; i < tensor.dimension(0); ++i) {
        std::cout << tensor(i) << " ";
    }
    std::cout << std::endl;
}

void print_2d_tensor(Tensor2dXf tensor) {
    std::cout << "Tensor elements (2D):" << std::endl;
    for (std::size_t i = 0; i < tensor.dimension(0); ++i) { // Iterate over rows
        for (std::size_t j = 0; j < tensor.dimension(1); ++j) { // Iterate over columns
            std::cout << tensor(i, j) << " "; // Print each element
        }
        std::cout << std::endl; // New line after each row
    }
}

Tensor1dXf TorchToEigen(const torch::Tensor& tensor){
    Tensor1dXf eigen_tensor(tensor.numel());
    for(size_t i=0;i<tensor.numel();i++){
        eigen_tensor(i)=tensor[i].item<float>();
    }
    return eigen_tensor;
}

bool TestIfEqual(Tensor1dXf tensor1, Tensor1dXf tensor2, float tolerance = 1e-5){
     if (tensor1.dimensions() != tensor2.dimensions()) {
        return false; 
    }
    for (int i = 0; i < tensor1.size(); ++i) {
        if (std::abs(tensor1(i) - tensor2(i)) > tolerance) {
            return false; 
        }
    }
    return true; 
}
//make the test more organized and create a test for all models
void TestSimpleModel(){
    bool testPased=1;
    fs::path base_path = "/home/torfinhell/Denoiser.cpp/tests/test_data/SimpleModel";
    fs::path input_path=base_path / "input.pth";
    fs::path model_path=base_path / "model.pth";
    fs::path prediction_path=base_path / "prediction.pth";
    torch::jit::script::Module prior_input, model, prior_prediction;
    torch::Tensor input_tensors, prediction_tensors;
    try {
        model = torch::jit::load(model_path);
        prior_input = torch::jit::load(input_path);
        prior_prediction = torch::jit::load(prediction_path);
        input_tensors = prior_input.attr("prior").toTensor();
        prediction_tensors = prior_prediction.attr("prior").toTensor();
        std::cout << "Simple Model loaded successfully"<<std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what()<<std::endl;
        testPased=0;
    }
    Tensor1dXf input=TorchToEigen(input_tensors);
    Tensor1dXf prediction=TorchToEigen(prediction_tensors);
    SimpleModel simple_model, loaded_model;
    if(!simple_model.load_from_jit_module(model)){
        std::cerr<<"Coudn't load model from jit."<<std::endl;
        testPased=0;
    }
    std::ios::sync_with_stdio(false);
    std::ofstream output_file(base_path / "data.txt");
    if (!output_file) {
        std::cerr << "Error opening file!" << std::endl;
        std::cout<<"Simple Model Test not passed"<<std::endl;
        return;
    }
    simple_model.load_to_file(output_file);
    if (!output_file.good()) {
        std::cerr << "Error writing to file!" << std::endl;
    }
    output_file.flush();
    output_file.close();
    std::ifstream input_file(base_path / "data.txt");
    if (!input_file) {
        std::cerr << "Error opening file!" << std::endl;
        std::cout<<"Simple Model Test not passed"<<std::endl;
        return;
    }
    loaded_model.load_from_file(input_file);
    input_file.close();
    if(!simple_model.IsEqual(loaded_model, 1e-5)){
        std::cerr<<"Model Is not correctly loaded. The Max difference between there elements"<<simple_model.MaxAbsDifference(loaded_model)<<std::endl;
        testPased=0;
    }  
    simple_model.forward(input);
    if(!TestIfEqual(prediction, simple_model.result)){
        std::cerr<<"Error Tensors in Simple Model test are not equal..."<<std::endl;
        testPased=0;
    }
    if(testPased){
        std::cout<<"Simple Model Test Successfully passed"<<std::endl;
    }else{
        std::cout<<"Simple Model Test not passed"<<std::endl;
    }
}
