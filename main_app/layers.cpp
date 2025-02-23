#include"layers.h"
#include "denoiser.h"
#include "tensors.h"
#include <array>
#include <fstream>
#include <stdexcept>
#include<cassert>
#include "tests.h"
#include <algorithm>
//write general function for equality(maxabs) for print tensor for writete nsorreadtensor
Eigen::array<Eigen::IndexPair<int>, 1>  product_dims_reg = { Eigen::IndexPair<int>(1, 0)};
Eigen::array<Eigen::IndexPair<int>, 1>  product_dims_sec_transposed = { Eigen::IndexPair<int>(1, 1)};
Eigen::array<Eigen::IndexPair<int>, 1>  product_dims_first_transposed = { Eigen::IndexPair<int>(0, 0)};
Eigen::array<Eigen::IndexPair<int>, 1>  product_dims_both_transposed = { Eigen::IndexPair<int>(0, 1)};
void SimpleModel::forward(Tensor1dXf tensor) {
    assert(tensor.dimension(0) == fc_weights.dimension(0)); 
    assert(fc_bias.dimension(0) == fc_weights.dimension(1));
    result = (tensor.contract(fc_weights,product_dims_first_transposed)) + fc_bias;
}
void writeTensor(Tensor1dXf tensor, std::ofstream& outputstream){
    size_t size=tensor.size();
    outputstream<<size<<" ";
    for(size_t i=0;i<size;i++){
        outputstream<<tensor(i)<<" ";
    }
}
void writeTensor(Tensor2dXf tensor, std::ofstream& outputstream){
    size_t dimension1=tensor.dimension(0), dimension2=tensor.dimension(1);
    outputstream<<dimension1<<" "<<dimension2<<" ";
    for(size_t i=0;i<dimension1;i++){
        for(size_t j=0;j<dimension2;j++){
            outputstream<<tensor(i,j)<<" ";
        }
    }
}
void readTensor(Tensor1dXf& tensor, std::ifstream& inputstream){
    size_t size;
    inputstream>>size;
    tensor.resize(size);
    for(size_t i=0;i<size;i++){
        inputstream>>tensor(i);
    }
}
void readTensor(Tensor2dXf& tensor, std::ifstream& inputstream){
    size_t dimension1, dimension2;
    inputstream>>dimension1>>dimension2;
    tensor.resize(static_cast<long>(dimension1), static_cast<long>(dimension2));
    for(size_t i=0;i<dimension1;i++){
        for(size_t j=0;j<dimension2;j++){
            inputstream>>tensor(i,j);
        }
    }
}
bool SimpleModel::load_from_jit_module(torch::jit::script::Module module) {
    try {
        for (const auto& param : module.named_parameters()) {
            // std::cout << "Param: " << param.name << ", Shape: " << param.value.sizes() << std::endl;

            if (param.name == "fc.weight") {
                if (param.value.dim() != 2) {
                    throw std::runtime_error("Expected fc.weight to be a 2D tensor");
                }
                fc_weights.resize(param.value.size(1), param.value.size(0));
                std::memcpy(fc_weights.data(), param.value.data_ptr<float>(), param.value.numel() * sizeof(float));
            } else if (param.name == "fc.bias") {
                if (param.value.dim() != 1) {
                    throw std::runtime_error("Expected fc.bias to be a 1D tensor");
                }
                fc_bias.resize(param.value.size(0));
                std::memcpy(fc_bias.data(), param.value.data_ptr<float>(), param.value.numel() * sizeof(float));
            } else {
                throw std::runtime_error("Module is not of fc.weight or fc.bias");
            }
        }
    } catch (const c10::Error& e) {
        std::cerr << "Error loading from JIT module: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        return false;
    }

    return true;
}
 

void SimpleModel::load_to_file(std::ofstream& outputstream) {
    writeTensor(fc_weights, outputstream);
    writeTensor(fc_bias, outputstream);
}

void SimpleModel::load_from_file(std::ifstream& inputstream) {
    readTensor(fc_weights, inputstream);
    readTensor(fc_bias, inputstream);
}

float SimpleModel::MaxAbsDifference(const SimpleModel& other) {
    Eigen::array<Eigen::IndexPair<int>, 1> dims = {Eigen::IndexPair<int>(1, 0)};
    Eigen::Tensor<float, 0> max_weight_tensor = (other.fc_weights.cast<float>() - fc_weights.cast<float>()).maximum();
    Eigen::Tensor<float, 0> max_bias_tensor = (other.fc_bias.cast<float>() - fc_bias.cast<float>()).maximum();
    float max_weight_diff = max_weight_tensor(0);
    float max_bias_diff = max_bias_tensor(0); 
    return std::max(max_bias_diff, max_weight_diff);
}

bool SimpleModel::IsEqual(const SimpleModel& other, float tolerance) {
    return MaxAbsDifference(other)<=tolerance;
}
