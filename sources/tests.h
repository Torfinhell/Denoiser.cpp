#include <Eigen/Dense>
#include <array>
#include <iostream>
#include <torch/torch.h>
#include <unsupported/Eigen/CXX11/Tensor>

void print_model_layers(const torch::jit::script::Module &module);

template <typename EigenTensor>
float MaxAbsDifference(EigenTensor tensor1, EigenTensor tensor2)
{
    assert(tensor1.dimensions() == tensor2.dimensions());
    Eigen::Tensor<float, 0> max_abs_tensor =
        (tensor2 - tensor1).abs().maximum();
    return max_abs_tensor(0);
}
template <typename EigenTensor>
bool TestIfEqual(EigenTensor tensor1, EigenTensor tensor2,
                 float tolerance = 1e-5)
{
    assert(tensor1.dimensions() == tensor2.dimensions());
    return MaxAbsDifference(tensor1, tensor2) <= tolerance;
}

template <typename EigenTensor, int NumDim>
EigenTensor TorchToEigen(const torch::Tensor &torch_tensor)
{
    assert(torch_tensor.dim() == NumDim);
    std::array<long, NumDim> dimensions;
    for (int i = 0; i < NumDim; ++i) {
        dimensions[i] = torch_tensor.size(i);
    }
    EigenTensor eigen_tensor(dimensions);
    std::array<long, NumDim> indices;
    for (int64_t i = 0; i < torch_tensor.numel(); ++i) {
        int64_t linear_index = i;
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
        for (int64_t i = 0; i < tensor.size(); i++) {
            std::cout << tensor(i) << " ";
        }
        std::cout << std::endl;
    }
    else if constexpr (NumDim == 2) {
        for (int64_t i = 0; i < tensor.dimension(0); i++) {
            for (int64_t j = 0; j < tensor.dimension(1); j++) {
                std::cout << tensor(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }
    else {
        if (current_pos == NumDim - 2) {
            std::cout << "(";
            for (int64_t i = 0; i < tensor.rank() - 2; i++) {
                std::cout << indices[i];
                std::cout << ",";
            }
            std::cout << "): \n";
            for (int64_t i = 0; i < tensor.dimension(tensor.rank() - 2); i++) {
                for (int64_t j = 0; j < tensor.dimension(tensor.rank() - 1);
                     j++) {
                    indices[tensor.rank() - 2] = i;
                    indices[tensor.rank() - 1] = j;
                    std::cout << tensor(indices) << " ";
                }
                std::cout << std::endl;
            }
        }
        else {
            for (int64_t i = 0; i < tensor.dimension(current_pos); i++) {
                indices[current_pos] = i;
                print_tensor<Tensor, NumDim>(tensor, indices, current_pos + 1);
            }
        }
    }
}

void TestSimpleModel();
void TestOneEncoder();

void TestSimpleEncoderDecoder();

void TestSimpleEncoderDecoderLSTM();

void TestDNS48();

void TestDemucsFullForwardAudio();

void TestStreamerForwardAudio();

void GenerateTestsforAudio();

void TestResampler();

void TestMP3();
