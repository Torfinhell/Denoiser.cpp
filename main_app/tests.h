#include "denoiser.h"
#include "layers.h"
#include "load_model.h"
#include "tensors.h"

#include <Eigen/Dense>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <torch/torch.h>
#include <unsupported/Eigen/CXX11/Tensor>

namespace fs = std::filesystem;
using namespace Eigen;

void print_model_layers(const torch::jit::script::Module &module);

template <typename EigenTensor>
float MaxAbsDifference(EigenTensor tensor1, EigenTensor tensor2)
{
    if (tensor1.dimensions() != tensor2.dimensions()) {
        throw std::runtime_error("Error Comparing Tensors of different size");
    }
    Eigen::Tensor<float, 0> max_abs_tensor =
        (tensor2 - tensor1).abs().maximum();
    float max_diff = max_abs_tensor(0);
    return max_diff;
    return 0;
}
template <typename EigenTensor>
bool TestIfEqual(EigenTensor tensor1, EigenTensor tensor2,
                 float tolerance = 1e-5)
{
    if (tensor1.dimensions() != tensor2.dimensions()) {
        throw std::runtime_error("Comparing Tensors of different size");
    }
    return MaxAbsDifference(tensor1, tensor2) <= tolerance;
}

template <typename EigenTensor, int NumDim>
EigenTensor TorchToEigen(const torch::Tensor &torch_tensor)
{
    std::array<long, NumDim> dimensions;
    for (int i = 0; i < NumDim; ++i) {
        dimensions[i] = torch_tensor.size(i);
    }
    EigenTensor eigen_tensor(dimensions);
    std::array<long, NumDim> indices;
    for (size_t i = 0; i < torch_tensor.numel(); ++i) {
        size_t linear_index = i;
        for (int d = NumDim - 1; d >= 0; --d) {
            indices[d] = linear_index % dimensions[d];
            linear_index /= dimensions[d];
        }
        eigen_tensor(indices) =
            torch_tensor.flatten()[i]
                .template item<typename EigenTensor::Scalar>();
    }

    return eigen_tensor;
}

template <typename Tensor, int NumDim>
void print_tensor(const Tensor &tensor, std::array<long, NumDim> &indices,
                  int current_pos = 0)
{
    if constexpr (NumDim == 1) {
        for (size_t i = 0; i < tensor.size(); i++) {
            std::cout << tensor(i) << " ";
        }
        std::cout << std::endl;
    }
    else if constexpr (NumDim == 2) {
        for (size_t i = 0; i < tensor.dimension(0); i++) {
            for (size_t j = 0; j < tensor.dimension(1); j++) {
                std::cout << tensor(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }
    else {
        if (current_pos == NumDim - 2) {
            std::cout << "(";
            for (size_t i = 0; i < tensor.rank() - 2; i++) {
                std::cout << indices[i];
                std::cout << ",";
            }
            std::cout << "): \n";
            for (size_t i = 0; i < tensor.dimension(tensor.rank() - 2); i++) {
                for (size_t j = 0; j < tensor.dimension(tensor.rank() - 1);
                     j++) {
                    indices[tensor.rank() - 2] = i;
                    indices[tensor.rank() - 1] = j;
                    std::cout << tensor(indices) << " ";
                }
                std::cout << std::endl;
            }
        }
        else {
            for (size_t i = 0; i < tensor.dimension(current_pos); i++) {
                indices[current_pos] = i;
                print_tensor<Tensor, NumDim>(tensor, current_pos + 1, indices);
            }
        }
    }
}

void TestSimpleModel();
