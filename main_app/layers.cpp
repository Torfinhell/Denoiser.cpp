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
template <typename T> T max_of_multiple(std::initializer_list<T> values)
{
    return *std::max_element(values.begin(), values.end());
}

void SimpleModel::forward(Tensor1dXf tensor)
{
    assert(tensor.dimension(0) == fc_weights.dimension(0));
    assert(fc_bias.dimension(0) == fc_weights.dimension(1));
    assert(tensor.size() > 0);
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

Tensor3dXf OneEncoder::forward(Tensor3dXf tensor)
{ // проверить точно ли такой тип
    auto res1 = conv_1_1d.forward(tensor, 1, hidden, kernel_size, stride);
    auto res2 = relu.forward(res1);
    auto res3 = conv_2_1d.forward(res2, hidden, hidden * ch_scale, 1);
    auto res4 = glu.forward(res3);
    return res4;
}

bool OneEncoder::load_from_jit_module(torch::jit::script::Module module)
{
    try {
        for (const auto &child : module.named_children()) {
            if (child.name == "encoder") {
                if (!conv_1_1d.load_from_jit_module(child.value, "0.0")) {
                    return false;
                }
                if (!conv_2_1d.load_from_jit_module(child.value, "0.2")) {
                    return false;
                }
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

void OneEncoder::load_to_file(std::ofstream &outputstream)
{
    conv_1_1d.load_to_file(outputstream);
    conv_2_1d.load_to_file(outputstream);
}

void OneEncoder::load_from_file(std::ifstream &inputstream)
{
    conv_1_1d.load_from_file(inputstream);
    conv_2_1d.load_from_file(inputstream);
}

float OneEncoder::MaxAbsDifference(const OneEncoder &other)
{
    return max_of_multiple({conv_1_1d.MaxAbsDifference(other.conv_1_1d),
                            conv_2_1d.MaxAbsDifference(other.conv_2_1d)});
}

bool OneEncoder::IsEqual(const OneEncoder &other, float tolerance)
{
    return MaxAbsDifference(other) <= tolerance;
}

Tensor3dXf Conv1D::forward(Tensor3dXf tensor, int InputChannels,
                           int OutputChannels, int kernel_size, int stride)
{
    int batch_size = tensor.dimension(0);
    int length = tensor.dimension(2);
    int new_length = GetSize(length, kernel_size, stride);
    assert(conv_weights.dimension(0) == OutputChannels);
    assert(conv_weights.dimension(1) == InputChannels);
    assert(conv_weights.dimension(2) == kernel_size);
    assert(tensor.dimension(1) == InputChannels);
    assert(kernel_size > 0 && kernel_size <= length);
    assert(stride > 0 && stride <= length);
    Tensor3dXf newtensor(batch_size, OutputChannels, new_length);
    newtensor.setZero();
    for (int channel = 0; channel < OutputChannels; channel++) {
        for (int batch = 0; batch < batch_size; batch++) {
            int counter = 0;
            for (int pos = 0; pos + kernel_size <= length;
                 pos += stride, counter++) {
                assert(counter < new_length);
                for (int i = 0; i < kernel_size; i++) {
                    for (int input_channel = 0; input_channel < InputChannels;
                         input_channel++) {
                        newtensor(batch, channel, counter) +=
                            tensor(batch, input_channel, pos + i) *
                            conv_weights(channel, input_channel, i);
                    }
                }
                newtensor(batch, channel, counter) += conv_bias(channel);
            }
            assert(counter == new_length);
        }
    }
    return newtensor;
}

bool Conv1D::load_from_jit_module(torch::jit::script::Module module,
                                  std::string weights_index = "0.0")
{
    try {
        for (const auto &param : module.named_parameters()) {
            if (param.name == weights_index + ".weight") {
                assert(param.value.dim() == 3);
                Tensor3dXf read_conv_weights(param.value.size(2),
                                             param.value.size(1),
                                             param.value.size(0)); // как точно
                std::memcpy(read_conv_weights.data(),
                            param.value.data_ptr<float>(),
                            param.value.numel() * sizeof(float));
                conv_weights =
                    read_conv_weights.shuffle(std::array<int, 3>{2, 1, 0});
            }
            else if (param.name == weights_index + ".bias") {
                assert(param.value.dim() == 1);
                conv_bias.resize(param.value.size(0));
                std::memcpy(conv_bias.data(), param.value.data_ptr<float>(),
                            param.value.numel() * sizeof(float));
            }
            else if (param.name.rfind(".bias") != param.name.size() - 5 &&
                     param.name.rfind(".weight") != param.name.size() - 7) {
                throw std::runtime_error(
                    "Conv1D has something else besides weight and bias: " +
                    param.name);
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
void Conv1D::load_to_file(std::ofstream &outputstream)
{
    WriteTensor<Tensor3dXf, 3>(conv_weights, outputstream, indices_dim_3);
    WriteTensor<Tensor1dXf, 1>(conv_bias, outputstream, indices_dim_1);
}

void Conv1D::load_from_file(std::ifstream &inputstream)
{
    ReadTensor<Tensor3dXf, 3>(conv_weights, inputstream, indices_dim_3);
    ReadTensor<Tensor1dXf, 1>(conv_bias, inputstream, indices_dim_1);
}

float Conv1D::MaxAbsDifference(const Conv1D &other)
{
    return max_of_multiple(
        {::MaxAbsDifference(conv_bias, other.conv_bias),
         ::MaxAbsDifference(conv_weights, other.conv_weights)});
}

bool Conv1D::IsEqual(const Conv1D &other, float tolerance)
{
    return MaxAbsDifference(other) <= tolerance;
}

int Conv1D::GetSize(int size, int kernel_size, int stride)
{
    assert(kernel_size <= size);
    assert(stride <= size);
    return (size - kernel_size) / stride + 1;
}

Tensor3dXf RELU::forward(Tensor3dXf tensor)
{
    auto func = [](float x) { return (x > 0.0f) ? x : 0.0f; };
    return tensor.unaryExpr(func);
}


Tensor3dXf GLU::forward(Tensor3dXf tensor)
{
    int batch_size = tensor.dimension(0);
    int channels = tensor.dimension(1);
    int height = tensor.dimension(2);
    
    if (channels % 2 != 0) {
        throw std::invalid_argument("Number of channels should be divisible by 2 for GLU 3D");
    }
    Eigen::array<Eigen::Index, 3> offset1 = {0, 0, 0};
    Eigen::array<Eigen::Index, 3> offset2 = {0, channels / 2, 0};
    Eigen::array<Eigen::Index, 3> extent = {batch_size, channels / 2, height};
    Tensor<float, 3> A = tensor.slice(offset1, extent);
    Tensor<float, 3> B = tensor.slice(offset2, extent);
    B = B.unaryExpr([](float x) { return 1.0f / (1.0f + std::exp(-x)); });
    return A * B;
}
