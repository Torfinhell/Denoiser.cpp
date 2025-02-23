#include "layers.h"
#include "denoiser.h"
#include "tensors.h"
#include "tests.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <fstream>
#include <stdexcept>
Eigen::array<Eigen::IndexPair<int>, 1> product_dims_reg = {
    Eigen::IndexPair<int>(1, 0)};
Eigen::array<Eigen::IndexPair<int>, 1> product_dims_sec_transposed = {
    Eigen::IndexPair<int>(1, 1)};
Eigen::array<Eigen::IndexPair<int>, 1> product_dims_first_transposed = {
    Eigen::IndexPair<int>(0, 0)};
Eigen::array<Eigen::IndexPair<int>, 1> product_dims_both_transposed = {
    Eigen::IndexPair<int>(0, 1)};
void SimpleModel::forward(Tensor1dXf tensor)
{
    assert(tensor.dimension(0) == fc_weights.dimension(0));
    assert(fc_bias.dimension(0) == fc_weights.dimension(1));
    result =
        (tensor.contract(fc_weights, product_dims_first_transposed)) + fc_bias;
}
template <typename EigenTensor, int NumDim>
void WriteTensor(EigenTensor &tensor, std::ofstream &outputstream,
                 std::array<long, NumDim> &indices, int current_pos = 0)
{
    if (current_pos == 0) {
        for (int i = 0; i < NumDim; i++) {
            outputstream << tensor.dimension(i) << " ";
        }
    }
    if (current_pos == NumDim) {
        outputstream << tensor(indices) << " ";
    }
    else {
        for (size_t i = 0; i < tensor.dimension(current_pos); i++) {
            indices[current_pos] = i;
            WriteTensor<EigenTensor, NumDim>(tensor, outputstream, indices,
                                             current_pos + 1);
        }
    }
}

template <typename EigenTensor, int NumDim>
void ReadTensor(EigenTensor &tensor, std::ifstream &inputstream,
                std::array<long, NumDim> &indices, int current_pos = 0)
{
    std::array<long, NumDim> dimensions;
    if (current_pos == 0) {
        for (int i = 0; i < NumDim; i++) {
            inputstream >> dimensions[i];
        }
        tensor.resize(dimensions);
    }
    if (current_pos == NumDim) {
        inputstream >> tensor(indices);
    }
    else {
        for (size_t i = 0; i < tensor.dimension(current_pos); i++) {
            indices[current_pos] = i;
            ReadTensor<EigenTensor, NumDim>(tensor, inputstream, indices,
                                            current_pos + 1);
        }
    }
}
bool SimpleModel::load_from_jit_module(torch::jit::script::Module module)
{
    try {
        for (const auto &param : module.named_parameters()) {
            if (param.name == "fc.weight") {
                assert(param.value.dim() == 2);
                fc_weights.resize(param.value.size(1), param.value.size(0));
                std::memcpy(fc_weights.data(), param.value.data_ptr<float>(),
                            param.value.numel() * sizeof(float));
            }
            else if (param.name == "fc.bias") {
                assert(param.value.dim() == 1);
                fc_bias.resize(param.value.size(0));
                std::memcpy(fc_bias.data(), param.value.data_ptr<float>(),
                            param.value.numel() * sizeof(float));
            }
            else {
                throw std::runtime_error(
                    "Module is not of fc.weight or fc.bias");
            }
        }
    }
    catch (const c10::Error &e) {
        std::cerr << "Error loading from JIT module: " << e.what() << std::endl;
        return false;
    }
    catch (const std::exception &e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        return false;
    }

    return true;
}

std::array<long, 1> indices_dim_1 = {0};
std::array<long, 2> indices_dim_2 = {0, 0};
std::array<long, 4> indices_dim_3 = {0, 0, 0};
std::array<long, 4> indices_dim_4 = {0, 0, 0, 0};
void SimpleModel::load_to_file(std::ofstream &outputstream)
{
    WriteTensor<Tensor2dXf, 2>(fc_weights, outputstream, indices_dim_2);
    WriteTensor<Tensor1dXf, 1>(fc_bias, outputstream, indices_dim_1);
}

void SimpleModel::load_from_file(std::ifstream &inputstream)
{
    ReadTensor<Tensor2dXf, 2>(fc_weights, inputstream, indices_dim_2);
    ReadTensor<Tensor1dXf, 1>(fc_bias, inputstream, indices_dim_1);
}

float SimpleModel::MaxAbsDifference(const SimpleModel &other)
{
    assert(fc_weights.size() == other.fc_weights.size() &&
           "Weights must be of the same size");
    assert(fc_bias.size() == other.fc_bias.size() &&
           "Biases must be of the same size");
    float weight_diff =
        ::MaxAbsDifference<Tensor2dXf>(other.fc_weights, fc_weights);
    float bias_diff = ::MaxAbsDifference<Tensor1dXf>(other.fc_bias, fc_bias);
    return std::max(weight_diff, bias_diff);
}

bool SimpleModel::IsEqual(const SimpleModel &other, float tolerance)
{
    return MaxAbsDifference(other) <= tolerance;
}
