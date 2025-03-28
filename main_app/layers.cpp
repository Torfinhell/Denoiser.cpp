#include "layers.h"
#include "Eigen/src/Core/util/Constants.h"
#include "Eigen/src/Core/util/Meta.h"
#include "audio.h"
#include "denoiser.h"
#include "tensors.h"
#include "tests.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorTraits.h"
#include "unsupported/Eigen/FFT"
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unsupported/Eigen/CXX11/ThreadPool>
#include <vector>
#define PI 3.141592653589793
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
{
    // Log the initial tensor shape
    // std::cout << "Input tensor shape: " << tensor.dimensions() << std::endl;

    // Measure time for conv_1_1d
    auto start = std::chrono::high_resolution_clock::now();
    auto res1 = conv_1_1d.forward(tensor, chin, hidden, kernel_size, stride);
    auto end = std::chrono::high_resolution_clock::now();
    // std::cout << "Time for conv_1_1d: "
    //           << std::chrono::duration_cast<std::chrono::microseconds>(end -
    //                                                                    start)
    //                  .count()
    //           << " microseconds" << std::endl;

    // Measure time for relu
    start = std::chrono::high_resolution_clock::now();
    auto res2 = relu.forward(res1);
    end = std::chrono::high_resolution_clock::now();
    // std::cout << "Time for relu: "
    //           << std::chrono::duration_cast<std::chrono::microseconds>(end -
    //                                                                    start)
    //                  .count()
    //           << " microseconds" << std::endl;
    // std::cout << "Input tensor shape in conv2: " << res2.dimensions() <<
    // std::endl; Measure time for conv_2_1d
    start = std::chrono::high_resolution_clock::now();
    auto res3 = conv_2_1d.forward(res2, hidden, hidden * ch_scale, 1);
    end = std::chrono::high_resolution_clock::now();
    // std::cout << "Time for conv_2_1d: "
    //           << std::chrono::duration_cast<std::chrono::microseconds>(end -
    //                                                                    start)
    //                  .count()
    //           << " microseconds" << std::endl;

    // Measure time for glu
    start = std::chrono::high_resolution_clock::now();
    auto res4 = glu.forward(res3);
    end = std::chrono::high_resolution_clock::now();
    // std::cout << "Time for glu: "
    //           << std::chrono::duration_cast<std::chrono::microseconds>(end -
    //                                                                    start)
    //                  .count()
    //           << " microseconds" << std::endl;

    return res4;
}

bool OneEncoder::load_from_jit_module(torch::jit::script::Module module,
                                      std::string weights_index)
{
    try {
        for (const auto &child : module.named_children()) {
            if (child.name == "encoder") {
                if (!conv_1_1d.load_from_jit_module(child.value,
                                                    weights_index + ".0")) {
                    return false;
                }
                if (!conv_2_1d.load_from_jit_module(child.value,
                                                    weights_index + ".2")) {
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

bool Conv1D::load_from_jit_module(torch::jit::script::Module module,
                                  std::string weights_index = "0.0")
{
    try {
        for (const auto &param : module.named_parameters()) {
            if (param.name == weights_index + ".weight") {
                assert(param.value.dim() == 3);
                Tensor3dXf read_conv_weights(param.value.size(2),
                                             param.value.size(1),
                                             param.value.size(0));
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
        throw std::invalid_argument(
            "Number of channels should be divisible by 2 for GLU 3D");
    }
    Eigen::array<Eigen::Index, 3> offset1 = {0, 0, 0};
    Eigen::array<Eigen::Index, 3> offset2 = {0, channels / 2, 0};
    Eigen::array<Eigen::Index, 3> extent = {batch_size, channels / 2, height};
    Tensor<float, 3> A = tensor.slice(offset1, extent);
    Tensor<float, 3> B = tensor.slice(offset2, extent);
    auto Sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    B = B.unaryExpr(Sigmoid);
    return A * B;
}

Tensor3dXf OneDecoder::forward(Tensor3dXf tensor)
{
    auto res1 = conv_1_1d.forward(tensor, hidden, hidden * ch_scale, 1, 1);
    auto res2 = glu.forward(res1);
    auto res3 = conv_tr_1_1d.forward(res2, hidden, chout, kernel_size, stride);
    return res3;
}

bool OneDecoder::load_from_jit_module(torch::jit::script::Module module,
                                      std::string weights_index)
{
    try {
        for (const auto &child : module.named_children()) {
            if (child.name == "decoder") {
                if (!conv_1_1d.load_from_jit_module(child.value,
                                                    weights_index + ".0")) {
                    return false;
                }
                if (!conv_tr_1_1d.load_from_jit_module(child.value,
                                                       weights_index + ".2")) {
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

void OneDecoder::load_to_file(std::ofstream &outputstream)
{
    conv_1_1d.load_to_file(outputstream);
    conv_tr_1_1d.load_to_file(outputstream);
}

void OneDecoder::load_from_file(std::ifstream &inputstream)
{
    conv_1_1d.load_from_file(inputstream);
    conv_tr_1_1d.load_from_file(inputstream);
}

float OneDecoder::MaxAbsDifference(const OneDecoder &other)
{
    return max_of_multiple<float>(
        {conv_1_1d.MaxAbsDifference(other.conv_1_1d),
         conv_tr_1_1d.MaxAbsDifference(other.conv_tr_1_1d)});
}

bool OneDecoder::IsEqual(const OneDecoder &other, float tolerance)
{
    return MaxAbsDifference(other) <= tolerance;
}

Tensor3dXf SimpleEncoderDecoder::forward(Tensor3dXf tensor)
{
    auto res1 = one_encoder.forward(tensor);
    auto res2 = one_decoder.forward(res1);
    return res2;
}

bool SimpleEncoderDecoder::load_from_jit_module(
    torch::jit::script::Module module)
{
    if (!one_encoder.load_from_jit_module(module)) {
        return false;
    }
    if (!one_decoder.load_from_jit_module(module)) {
        return false;
    }
    return true;
}

void SimpleEncoderDecoder::load_to_file(std::ofstream &outputstream)
{
    one_encoder.load_to_file(outputstream);
    one_decoder.load_to_file(outputstream);
}

void SimpleEncoderDecoder::load_from_file(std::ifstream &inputstream)
{
    one_encoder.load_from_file(inputstream);
    one_decoder.load_from_file(inputstream);
}

float SimpleEncoderDecoder::MaxAbsDifference(const SimpleEncoderDecoder &other)
{
    return max_of_multiple({one_encoder.MaxAbsDifference(other.one_encoder),
                            one_decoder.MaxAbsDifference(other.one_decoder)});
}

bool SimpleEncoderDecoder::IsEqual(const SimpleEncoderDecoder &other,
                                   float tolerance)
{
    return MaxAbsDifference(other) <= tolerance;
}

Tensor3dXf ConvTranspose1d::forward(Tensor3dXf tensor, int InputChannels,
                                    int OutputChannels, int kernel_size,
                                    int stride)
{
    int batch_size = tensor.dimension(0);
    int length = tensor.dimension(2);
    long new_length = GetTransposedSize(length, kernel_size, stride);
    assert(conv_tr_weights.dimension(0) == InputChannels);
    assert(conv_tr_weights.dimension(1) == OutputChannels);
    assert(conv_tr_weights.dimension(2) == kernel_size);
    assert(tensor.dimension(1) == InputChannels);
    assert(kernel_size > 0 && kernel_size <= new_length);
    assert(stride > 0);
    //we assume that kernel_size==stride*2
    // if (kernel_size == stride * 2) {
    auto func_get_ans = [batch_size, OutputChannels, length, stride](
                            const Tensor3dXf &input,
                            const Tensor3dXf &stride_kernel,
                            int padding_left = 0, int padding_right = 0) {
        Tensor4dXf newtensor = input.contract(
            stride_kernel, Eigen::array<Eigen::IndexPair<int>, 1>{
                                {Eigen::IndexPair<int>(1, 0)}});
        int new_length_for_stride = stride * length;
        Tensor4dXf other_tensor =
            newtensor.shuffle(std::array<long long, 4>{0, 2, 1, 3});
        Tensor3dXf res = other_tensor.reshape(std::array<long, 3>{
            batch_size, OutputChannels, new_length_for_stride});
        Eigen::array<std::pair<int, int>, 3> paddings;
        paddings[0] = std::make_pair(0, 0);
        paddings[1] = std::make_pair(0, 0);
        paddings[2] = std::make_pair(padding_left, padding_right);
        return res.pad(paddings);
    };
    std::array<long, 3> offset = {0, 0, 0};
    std::array<long, 3> offset1 = {0, 0, stride};
    std::array<long, 3> extent = {InputChannels, OutputChannels, stride};
    Tensor3dXf output_tensor =
        func_get_ans(tensor, conv_tr_weights.slice(offset, extent), 0,
                        stride) +
        func_get_ans(tensor, conv_tr_weights.slice(offset1, extent), stride,
                        0);
    output_tensor +=
        conv_tr_bias.reshape(std::array<long, 3>{1, OutputChannels, 1})
            .broadcast(std::array<long long, 3>{batch_size, 1, new_length});
    return output_tensor;
    // }
    // else if (kernel_size == stride) {
    //     Tensor4dXf newtensor = tensor.contract(
    //         conv_tr_weights, Eigen::array<Eigen::IndexPair<int>, 1>{
    //                              {Eigen::IndexPair<int>(1, 0)}});
    //     Tensor4dXf other_tensor =
    //         newtensor.shuffle(std::array<long long, 4>{0, 2, 1, 3});
    //     Tensor3dXf output_tensor = other_tensor.reshape(
    //         std::array<long, 3>{batch_size, OutputChannels, new_length});
    //     output_tensor +=
    //         conv_tr_bias.reshape(std::array<long, 3>{1, OutputChannels, 1})
    //             .broadcast(std::array<long long, 3>{batch_size, 1, new_length});
    //     return output_tensor;
    // }
    // else {
    //     Tensor3dXf newtensor(batch_size, OutputChannels, new_length);
    //     newtensor.setZero();

    //     for (int channel = 0; channel < OutputChannels; channel++) {
    //         for (int batch = 0; batch < batch_size; batch++) {
    //             for (int pos = 0; pos < length; pos++) {
    //                 for (int i = 0; i < kernel_size; i++) {
    //                     int output_pos = pos * stride + i;
    //                     if (output_pos < new_length) {
    //                         for (int input_channel = 0;
    //                              input_channel < InputChannels;
    //                              input_channel++) {
    //                             newtensor(batch, channel, output_pos) +=
    //                                 tensor(batch, input_channel, pos) *
    //                                 conv_tr_weights(input_channel, channel, i);
    //                         }
    //                     }
    //                 }
    //             }
    //             for (int pos = 0; pos < new_length; pos++) {
    //                 newtensor(batch, channel, pos) += conv_tr_bias(channel);
    //             }
    //         }
    //     }
    //     return newtensor;
    // }
}

bool ConvTranspose1d::load_from_jit_module(torch::jit::script::Module module,
                                           std::string weights_index)
{
    try {
        for (const auto &param : module.named_parameters()) {
            if (param.name == weights_index + ".weight") {
                assert(param.value.dim() == 3);
                Tensor3dXf read_conv_tr_weights(param.value.size(2),
                                                param.value.size(1),
                                                param.value.size(0));
                std::memcpy(read_conv_tr_weights.data(),
                            param.value.data_ptr<float>(),
                            param.value.numel() * sizeof(float));
                conv_tr_weights =
                    read_conv_tr_weights.shuffle(std::array<int, 3>{2, 1, 0});
            }
            else if (param.name == weights_index + ".bias") {
                assert(param.value.dim() == 1);
                conv_tr_bias.resize(param.value.size(0));
                std::memcpy(conv_tr_bias.data(), param.value.data_ptr<float>(),
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

void ConvTranspose1d::load_to_file(std::ofstream &outputstream)
{
    WriteTensor<Tensor3dXf, 3>(conv_tr_weights, outputstream, indices_dim_3);
    WriteTensor<Tensor1dXf, 1>(conv_tr_bias, outputstream, indices_dim_1);
}

void ConvTranspose1d::load_from_file(std::ifstream &inputstream)
{
    ReadTensor<Tensor3dXf, 3>(conv_tr_weights, inputstream, indices_dim_3);
    ReadTensor<Tensor1dXf, 1>(conv_tr_bias, inputstream, indices_dim_1);
}

float ConvTranspose1d::MaxAbsDifference(const ConvTranspose1d &other)
{
    return max_of_multiple(
        {::MaxAbsDifference(conv_tr_bias, other.conv_tr_bias),
         ::MaxAbsDifference(conv_tr_weights, other.conv_tr_weights)});
}

bool ConvTranspose1d::IsEqual(const ConvTranspose1d &other, float tolerance)
{
    return MaxAbsDifference(other) <= tolerance;
}

int ConvTranspose1d::GetTransposedSize(int size, int kernel_size, int stride)
{
    return stride * (size - 1) + kernel_size;
}

Tensor3dXf SimpleEncoderDecoderLSTM::forward(Tensor3dXf tensor)
{
    LstmState lstm_state;
    auto res1 = one_encoder.forward(tensor);
    auto res2 = lstm1.forward(res1.shuffle(std::array<long long, 3>{2, 0, 1}),
                              hidden, lstm_state);
    lstm_state.is_created = false;
    auto res3 = lstm2.forward(res2, hidden, lstm_state);
    auto res4 =
        one_decoder.forward(res3.shuffle(std::array<long long, 3>{1, 2, 0}));
    return res4;
}

bool SimpleEncoderDecoderLSTM::load_from_jit_module(
    torch::jit::script::Module module)
{
    if (!one_encoder.load_from_jit_module(module)) {
        return false;
    }
    if (!lstm1.load_from_jit_module(module, "0")) {
        return false;
    }
    if (!lstm2.load_from_jit_module(module, "1")) {
        return false;
    }
    if (!one_decoder.load_from_jit_module(module)) {
        return false;
    }
    return true;
}

void SimpleEncoderDecoderLSTM::load_to_file(std::ofstream &outputstream)
{
    one_encoder.load_to_file(outputstream);
    lstm1.load_to_file(outputstream);
    lstm2.load_to_file(outputstream);
    one_decoder.load_to_file(outputstream);
}

void SimpleEncoderDecoderLSTM::load_from_file(std::ifstream &inputstream)
{
    one_encoder.load_from_file(inputstream);
    lstm1.load_from_file(inputstream);
    lstm2.load_from_file(inputstream);
    one_decoder.load_from_file(inputstream);
}

float SimpleEncoderDecoderLSTM::MaxAbsDifference(
    const SimpleEncoderDecoderLSTM &other)
{
    return max_of_multiple({one_encoder.MaxAbsDifference(other.one_encoder),
                            one_decoder.MaxAbsDifference(other.one_decoder),
                            lstm1.MaxAbsDifference(other.lstm1),
                            lstm2.MaxAbsDifference(other.lstm2)});
}

bool SimpleEncoderDecoderLSTM::IsEqual(const SimpleEncoderDecoderLSTM &other,
                                       float tolerance)
{
    return MaxAbsDifference(other) <= tolerance;
}

Tensor3dXf OneLSTM::forward(Tensor3dXf tensor, int HiddenSize,
                            LstmState &lstm_state, bool bi)
{
    int length = tensor.dimension(0);
    int batch_size = tensor.dimension(1);
    int input_size = tensor.dimension(2);
    Tensor3dXf output(length, batch_size, HiddenSize);
    if (!lstm_state.is_created) {
        lstm_state.hidden_state.resize(
            std::array<long, 2>{HiddenSize, batch_size});
        lstm_state.cell_state.resize(
            std::array<long, 2>{HiddenSize, batch_size});
        lstm_state.hidden_state.setZero();
        lstm_state.cell_state.setZero();
        lstm_state.is_created = true;
    }

    auto ExtendColumn = [](Tensor2dXf column_weight,
                           long columns_count) -> Tensor2dXf {
        assert(column_weight.dimension(1) == 1);
        Tensor2dXf new_column_weight(column_weight.dimension(0), columns_count);
        for (long col = 0; col < columns_count; col++) {
            new_column_weight.chip(col, 1) = column_weight.chip(0, 1);
        }
        return column_weight;
    };
    auto tanh_func = [](float x) { return std::tanh(x); };
    auto sigmoid_func = [](float x) { return 1 / (1 + std::exp(-x)); };
    assert(lstm_weight_hh.dimension(0) == 4 * HiddenSize &&
           lstm_weight_hh.dimension(1) == HiddenSize);
    assert(lstm_weight_ih.dimension(0) == 4 * HiddenSize &&
           lstm_weight_ih.dimension(1) == input_size);
    assert(lstm_bias_ih.dimension(0) == 4 * HiddenSize &&
           lstm_bias_ih.dimension(1) == 1);
    assert(lstm_bias_hh.dimension(0) == 4 * HiddenSize &&
           lstm_bias_hh.dimension(1) == 1);
    for (int t = 0; t < length; t++) {
        Tensor2dXf x_t = tensor.chip(t, 0); //(input_size, batch_size)
        Tensor2dXf combined_input =
            lstm_weight_ih.contract(x_t, product_dims_sec_transposed) +
            ExtendColumn(lstm_bias_ih, batch_size);
        Tensor2dXf combined_hidden =
            lstm_weight_hh.contract(lstm_state.hidden_state, product_dims_reg) +
            ExtendColumn(lstm_bias_hh, batch_size);
        Tensor2dXf gates = combined_input + combined_hidden;
        std::array<long, 2> offset = {0, 0};
        std::array<long, 2> extent = {HiddenSize, batch_size};
        Tensor2dXf i_t = gates.slice(offset, extent).unaryExpr(sigmoid_func);
        offset[0] += HiddenSize;
        Tensor2dXf f_t = gates.slice(offset, extent).unaryExpr(sigmoid_func);
        offset[0] += HiddenSize;
        Tensor2dXf c_t = gates.slice(offset, extent).unaryExpr(tanh_func);
        offset[0] += HiddenSize;
        Tensor2dXf o_t = gates.slice(offset, extent).unaryExpr(sigmoid_func);
        lstm_state.cell_state = f_t * lstm_state.cell_state + i_t * c_t;
        lstm_state.hidden_state =
            o_t * lstm_state.cell_state.unaryExpr(tanh_func);
        output.chip(t, 0) =
            lstm_state.hidden_state.shuffle(std::array<long, 2>{1, 0});
    }
    return output;
}

bool OneLSTM::load_from_jit_module(torch::jit::script::Module module,
                                   std::string weights_index)
{
    try {
        for (const auto &param : module.named_parameters()) {
            if (param.name == "lstm.lstm.weight_ih_l" + weights_index) {
                assert(param.value.dim() == 2);
                Tensor2dXf read_lstm_weight_ih(param.value.size(1),
                                               param.value.size(0));
                std::memcpy(read_lstm_weight_ih.data(),
                            param.value.data_ptr<float>(),
                            param.value.numel() * sizeof(float));
                lstm_weight_ih =
                    read_lstm_weight_ih.shuffle(std::array<int, 2>{1, 0});
            }
            else if (param.name == "lstm.lstm.weight_hh_l" + weights_index) {
                assert(param.value.dim() == 2);
                Tensor2dXf read_lstm_weight_hh(param.value.size(1),
                                               param.value.size(0));
                std::memcpy(read_lstm_weight_hh.data(),
                            param.value.data_ptr<float>(),
                            param.value.numel() * sizeof(float));
                lstm_weight_hh =
                    read_lstm_weight_hh.shuffle(std::array<int, 2>{1, 0});
            }
            else if (param.name == "lstm.lstm.bias_ih_l" + weights_index) {
                assert(param.value.dim() == 1);
                lstm_bias_ih.resize(param.value.size(0), 1);
                std::memcpy(lstm_bias_ih.data(), param.value.data_ptr<float>(),
                            param.value.numel() * sizeof(float));
            }
            else if (param.name == "lstm.lstm.bias_hh_l" + weights_index) {
                assert(param.value.dim() == 1);
                lstm_bias_hh.resize(param.value.size(0), 1);
                std::memcpy(lstm_bias_hh.data(), param.value.data_ptr<float>(),
                            param.value.numel() * sizeof(float));
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

void OneLSTM::load_to_file(std::ofstream &outputstream)
{
    WriteTensor<Tensor2dXf, 2>(lstm_weight_ih, outputstream, indices_dim_2);
    WriteTensor<Tensor2dXf, 2>(lstm_weight_hh, outputstream, indices_dim_2);
    WriteTensor<Tensor2dXf, 2>(lstm_bias_ih, outputstream, indices_dim_2);
    WriteTensor<Tensor2dXf, 2>(lstm_bias_hh, outputstream, indices_dim_2);
}

void OneLSTM::load_from_file(std::ifstream &inputstream)
{
    ReadTensor<Tensor2dXf, 2>(lstm_weight_ih, inputstream, indices_dim_2);
    ReadTensor<Tensor2dXf, 2>(lstm_weight_hh, inputstream, indices_dim_2);
    ReadTensor<Tensor2dXf, 2>(lstm_bias_ih, inputstream, indices_dim_2);
    ReadTensor<Tensor2dXf, 2>(lstm_bias_hh, inputstream, indices_dim_2);
}

float OneLSTM::MaxAbsDifference(const OneLSTM &other)
{
    return max_of_multiple<float>({
        ::MaxAbsDifference(lstm_weight_ih, other.lstm_weight_ih),
        ::MaxAbsDifference(lstm_weight_hh, other.lstm_weight_hh),
        ::MaxAbsDifference(lstm_bias_ih, other.lstm_bias_ih),
        ::MaxAbsDifference(lstm_bias_hh, other.lstm_bias_hh),
    });
}

bool OneLSTM::IsEqual(const OneLSTM &other, float tolerance)
{
    return MaxAbsDifference(other) <= tolerance;
}

template <typename Tensor, int NumDim>
void GetMean(Tensor &tensor, Tensor &result, int dim,
             std::array<long, NumDim> arr, int pos = 0,
             bool is_first_call = true)
{
    if (is_first_call) {
        std::array<long, NumDim> dimensions = {};
        for (int dimension = 0; dimension < NumDim; dimension++) {
            dimensions[dimension] = tensor.dimension(dimension);
        }
        dimensions[dim] = 1;
        result.resize(dimensions);
        result.setZero();
    }

    if (pos == NumDim) {
        std::array<long, NumDim> arr_copy = arr;
        double sum = 0;
        for (int ind = 0; ind < tensor.dimension(dim); ind++) {
            arr[dim] = ind;
            sum += tensor(arr);
        }
        result(arr_copy) = sum / tensor.dimension(dim);
    }
    else if (pos == dim) {
        arr[dim] = 0;
        GetMean<Tensor, NumDim>(tensor, result, dim, arr, pos + 1, false);
    }
    else {
        for (int ind = 0; ind < tensor.dimension(pos); ind++) {
            arr[pos] = ind;
            GetMean<Tensor, NumDim>(tensor, result, dim, arr, pos + 1, false);
        }
    }
}
template <typename Tensor, int NumDim>
void GetStd(Tensor &tensor, Tensor &result, int dim,
            std::array<long, NumDim> arr, int pos = 0,
            bool is_first_call = true)
{
    if (is_first_call) {
        std::array<long, NumDim> dimensions = {};
        for (int dimension = 0; dimension < NumDim; dimension++) {
            dimensions[dimension] = tensor.dimension(dimension);
        }
        dimensions[dim] = 1;
        result.resize(dimensions);
        result.setZero();
    }

    if (pos == NumDim) {
        std::array<long, NumDim> arr_copy = arr;
        double mean = 0;
        long count = tensor.dimension(dim);
        for (int ind = 0; ind < count; ind++) {
            arr[dim] = ind;
            mean += tensor(arr);
        }
        mean /= count;
        for (int ind = 0; ind < count; ind++) {
            arr[dim] = ind;
            double diff = tensor(arr) - mean;
            result(arr_copy) += diff * diff;
        }
        if (count > 1) {
            result(arr_copy) /= (count - 1);
            result(arr_copy) = sqrt(result(arr_copy));
        }
        else {
            result(arr_copy) = 0;
        }
    }
    else if (pos == dim) {
        arr[dim] = 0;
        GetStd<Tensor, NumDim>(tensor, result, dim, arr, pos + 1, false);
    }
    else {
        for (int ind = 0; ind < tensor.dimension(pos); ind++) {
            arr[pos] = ind;
            GetStd<Tensor, NumDim>(tensor, result, dim, arr, pos + 1, false);
        }
    }
}

Tensor3dXf KernelUpsample2(int zeros = 56)
{
    int size = (4 * zeros + 1);
    Tensor1dXf window(size);
    for (int i = 0; i < size; i++) {
        window(i) = 0.5 * (1 - std::cos(2 * M_PI * i / (size - 1)));
    }

    int odd_size = 2 * zeros;
    Tensor1dXf window_odd(odd_size), t(odd_size);
    for (int i = 0; i < odd_size; i++) {
        window_odd(i) = window(1 + 2 * i);
        double a = -static_cast<double>(zeros) + 0.5;
        double b = static_cast<double>(zeros) - 0.5;
        t(i) = a + (b - a) * (static_cast<double>(i) / (odd_size - 1));
        t(i) = t(i) * M_PI;
    }
    auto sinc_func = [](float x) { return (x == 0) ? 1.0f : std::sin(x) / x; };
    Tensor1dXf kernel = (t.unaryExpr(sinc_func)) * window_odd;
    Tensor3dXf kernel_reshaped =
        kernel.reshape(std::array<long, 3>{1, 1, kernel.size()});
    return kernel_reshaped;
}
template <typename Tensor, typename TensorMore, int NumDim>
Tensor UpSample(Tensor tensor, std::array<long, NumDim> arr, int zeros = 56)
{
    long first_shape_size = 1;
    int last_dimension = tensor.dimension(NumDim - 1);
    for (long i = 0; i < NumDim - 1; i++) {
        first_shape_size *= tensor.dimension(i);
    }
    Tensor kernel = KernelUpsample2(zeros);
    Conv1D conv;
    conv.conv_weights = kernel;
    conv.conv_bias.resize(std::array<long, 1>{1});
    conv.conv_bias.setZero();
    Tensor3dXf res = conv.forward(tensor.reshape(std::array<long, 3>{
                                      first_shape_size, 1, last_dimension}),
                                  1, 1, kernel.dimension(2), 1, zeros);
    std::array<long, NumDim> extent;
    for (long i = 0; i < NumDim; i++) {
        extent[i] = res.dimension(i);
    }
    extent[NumDim - 1]--;
    std::array<long, NumDim> offset = {};
    offset[NumDim - 1] = 1;
    Tensor out = res.slice(offset, extent).reshape(tensor.dimensions());
    std::array<long, NumDim + 1> new_shape;
    for (int i = 0; i < NumDim; i++) {
        new_shape[i] = tensor.dimension(i);
    }
    new_shape[NumDim] = new_shape[NumDim - 1];
    new_shape[NumDim - 1] = 2;
    TensorMore stacked(new_shape);
    stacked.chip(0, NumDim - 1) = tensor;
    stacked.chip(1, NumDim - 1) = out;
    new_shape[NumDim - 1] = 2 * new_shape[NumDim];
    new_shape[NumDim] = 1;
    TensorMore stacked_reshaped = stacked.reshape(new_shape);
    Tensor y = stacked_reshaped.chip(0, NumDim);
    return y;
}

template <typename Tensor, typename TensorMore, int NumDim>
Tensor DownSample(Tensor tensor, std::array<long, NumDim> arr, int zeros = 56)
{
    Tensor3dXf padded_tensor = tensor;
    if (tensor.dimension(NumDim - 1) % 2 != 0) {
        Eigen::array<std::pair<int, int>, 3> paddings = {};
        paddings[2] = std::make_pair(0, 1);
        padded_tensor = tensor.pad(paddings);
    }
    std::array<long, NumDim> new_shape = padded_tensor.dimensions();
    new_shape[NumDim - 1] /= 2;

    Tensor3dXf tensor_even(new_shape), tensor_odd(new_shape);
    for (int i = 0; i < padded_tensor.dimension(NumDim - 1); i++) {
        if (i % 2 == 0) {
            tensor_even.chip(i / 2, NumDim - 1) =
                padded_tensor.chip(i, NumDim - 1);
        }
        else {
            tensor_odd.chip((i - 1) / 2, NumDim - 1) =
                padded_tensor.chip(i, NumDim - 1);
        }
    }
    long first_shape_size = 1,
         last_dimension = tensor_odd.dimension(NumDim - 1);
    for (long i = 0; i < NumDim - 1; i++) {
        first_shape_size *= tensor_odd.dimension(i);
    }
    Tensor kernel = KernelUpsample2(zeros);
    Conv1D conv;
    conv.conv_weights = kernel;
    conv.conv_bias.resize(std::array<long, 1>{1});
    conv.conv_bias.setZero();
    Tensor3dXf res = conv.forward(tensor_odd.reshape(std::array<long, 3>{
                                      first_shape_size, 1, last_dimension}),
                                  1, 1, kernel.dimension(2), 1, zeros);
    std::array<long, NumDim> extent = res.dimensions();
    extent[NumDim - 1]--;
    std::array<long, NumDim> offset = {};
    Tensor out = res.slice(offset, extent).reshape(tensor_odd.dimensions());
    return (tensor_even + out).unaryExpr([](float x) { return x / 2; });
}

Tensor3dXf DemucsModel::forward(Tensor3dXf mix, LstmState &lstm_state)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto [res1, skips, length, std_constant] = EncoderWorker(mix);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken for EncoderWorker: "
              << std::chrono::duration<double>(end - start).count()
              << " seconds" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    auto res2 = LSTMWorker(res1, lstm_state);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken for LSTMWorker: "
              << std::chrono::duration<double>(end - start).count()
              << " seconds" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    auto res3 = DecoderWorker(res2, skips, length, std_constant);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken for DecoderWorker: "
              << std::chrono::duration<double>(end - start).count()
              << " seconds" << std::endl;
    return res3;
}
std::tuple<Tensor3dXf, std::vector<Tensor3dXf>, int, float>
DemucsModel::EncoderWorker(Tensor3dXf mix)
{
    Tensor3dXf mono, std;
    GetMean<Tensor3dXf, 3>(mix, mono, 1, indices_dim_3);
    GetStd<Tensor3dXf, 3>(mono, std, 2, indices_dim_3);
    float std_constant = std(std::array<long, 3>{0, 0, 0});
    mix = mix.unaryExpr([std_constant](float x) {
        return x / (static_cast<float>(std_constant + 1e-3));
    });
    int length = mix.dimension(mix.NumDimensions - 1);
    Tensor3dXf x = mix;
    Eigen::array<std::pair<int, int>, 3> paddings = {};
    paddings[2] = std::make_pair(0, valid_length(length) - length);
    Tensor3dXf padded_x = x.pad(paddings);
    x = padded_x;
    std::vector<Tensor3dXf> skips;
    x = UpSample<Tensor3dXf, Tensor4dXf, 3>(x, indices_dim_3);
    x = UpSample<Tensor3dXf, Tensor4dXf, 3>(x, indices_dim_3);
    for (int i = 0; i < depth; i++) {
        x = encoders[i].forward(x);
        skips.push_back(x);
    }
    return {x, skips, length, std_constant};
}
Tensor3dXf DemucsModel::DecoderWorker(Tensor3dXf res2,
                                      std::vector<Tensor3dXf> &skips,
                                      int length, float std_constant)
{
    auto res3 = res2.shuffle(std::array<long long, 3>{1, 2, 0});
    Tensor3dXf x = res3;
    RELU relu;
    std::array<long, 3> offset = {};
    std::array<long, 3> extent;
    for (int i = 0; i < depth; i++) {
        Tensor3dXf skip = skips[depth - 1 - i];
        extent = skip.dimensions();
        extent[2] = x.dimension(2);
        x = x + skip.slice(offset, extent);
        x = decoders[i].forward(x);
        if (i != depth - 1) {
            x = relu.forward(x);
        }
    }
    x = DownSample<Tensor3dXf, Tensor4dXf, 3>(x, indices_dim_3);
    x = DownSample<Tensor3dXf, Tensor4dXf, 3>(x, indices_dim_3);
    extent = x.dimensions();
    extent[2] = length;
    Tensor3dXf x_slice = x.slice(offset, extent);
    return x_slice.unaryExpr(
        [std_constant](float x) { return std_constant * x; });
}
Tensor3dXf DemucsModel::LSTMWorker(Tensor3dXf x, LstmState &lstm_state)
{
    auto res1 = lstm1.forward(x.shuffle(std::array<long long, 3>{2, 0, 1}),
                              lstm_hidden, lstm_state);
    lstm_state.is_created = false;
    auto res2 = lstm2.forward(res1, lstm_hidden, lstm_state);
    return res2;
}
int DemucsModel::valid_length(int length)
{
    length = length * resample;
    for (int i = 0; i < depth; i++) {
        length = static_cast<int>(std::ceil((length - kernel_size) /
                                            static_cast<double>(stride))) +
                 1;
        length = std::max(length, 1);
    }
    for (int i = 0; i < depth; i++) {
        length = (length - 1) * stride + kernel_size;
    }
    length = static_cast<int>(std::ceil(length / resample));
    return length;
}

bool DemucsModel::load_from_jit_module(torch::jit::script::Module module)
{
    for (int i = 0; i < depth; i++) {
        if (!encoders[i].load_from_jit_module(module, std::to_string(i))) {
            return false;
        }
        if (!decoders[i].load_from_jit_module(module, std::to_string(i))) {
            return false;
        }
    }
    if (!lstm1.load_from_jit_module(module, "0")) {
        return false;
    }
    if (!lstm2.load_from_jit_module(module, "1")) {
        return false;
    }
    return true;
}

void DemucsModel::load_to_file(std::ofstream &outputstream)
{
    for (int i = 0; i < depth; i++) {
        encoders[i].load_to_file(outputstream);
        decoders[i].load_to_file(outputstream);
    }
    lstm1.load_to_file(outputstream);
    lstm2.load_to_file(outputstream);
}

void DemucsModel::load_from_file(std::ifstream &inputstream)
{
    for (int i = 0; i < depth; i++) {
        encoders[i].load_from_file(inputstream);
        decoders[i].load_from_file(inputstream);
    }
    lstm1.load_from_file(inputstream);
    lstm2.load_from_file(inputstream);
}

float DemucsModel::MaxAbsDifference(const DemucsModel &other)
{
    float max = 0;
    for (int i = 0; i < depth; i++) {
        max = max_of_multiple(
            {max, encoders[i].MaxAbsDifference(other.encoders[i]),
             decoders[i].MaxAbsDifference(other.decoders[i])});
    }
    return max_of_multiple({max, lstm1.MaxAbsDifference(other.lstm1),
                            lstm2.MaxAbsDifference(lstm2)});
}

bool DemucsModel::IsEqual(const DemucsModel &other, float tolerance)
{
    return MaxAbsDifference(other) <= tolerance;
}

Tensor2dXf DemucsStreamer::forward(Tensor2dXf wav)
{
    DemucsModel demucs_model;
    fs::path base_path = "../tests/test_data/dns48";
    std::ifstream input_file(base_path / "data.txt");
    demucs_model.load_from_file(input_file);
    const int resample_lookahead = 64;
    const int stride = 4;
    const int depth = 5;
    const int total_stride = stride * depth;
    const int num_frames = 1;
    int frame_length =
        demucs_model.valid_length(1) + total_stride * (num_frames - 1);
    const int total_length = frame_length + resample_lookahead;
    const int numThreads = 12;
    const int resample_buffer = 256;
    std::vector<Tensor3dXf> buffer_audio =
        SplitAudio(wav, total_length, total_stride);
    std::vector<Tensor3dXf> return_audio(buffer_audio.size());
    const int batch_size = 2;
    LstmState lstm_state;
    std::vector<std::thread> threads_encoders;
    std::vector<std::thread> threads_decoders;
    std::vector<std::tuple<Tensor3dXf, std::vector<Tensor3dXf>, int, float>>
        encoders_result(buffer_audio.size());
    std::vector<Tensor3dXf> lstm_result(buffer_audio.size());
    std::vector<Tensor3dXf> decoders_result(buffer_audio.size());
    auto start = std::chrono::high_resolution_clock::now();
    for (int start_thread = 0; start_thread < buffer_audio.size();
         start_thread += numThreads) {
        for (int i = start_thread;
             i < std::min(static_cast<int>(buffer_audio.size()),
                          start_thread + numThreads);
             i++) {
            threads_encoders.emplace_back([&, i]() {
                encoders_result[i] =
                    demucs_model.EncoderWorker(buffer_audio[i]);
            });
        }
        for (auto &t : threads_encoders) {
            t.join();
        }
        for (int i = start_thread;
             i < std::min(static_cast<int>(buffer_audio.size()),
                          start_thread + numThreads);
             i++) {
            lstm_result[i] = demucs_model.LSTMWorker(
                std::get<0>(encoders_result[i]), lstm_state);
        }
        for (int i = start_thread;
             i < std::min(static_cast<int>(buffer_audio.size()),
                          start_thread + numThreads);
             i++) {
            threads_decoders.emplace_back([&, i]() {
                decoders_result[i] = demucs_model.DecoderWorker(
                    lstm_result[i], std::get<1>(encoders_result[i]),
                    std::get<2>(encoders_result[i]),
                    std::get<3>(encoders_result[i]));
            });
        }
        for (auto &t : threads_decoders) {
            t.join();
        }
        threads_decoders.clear();
        threads_encoders.clear();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << ", Time taken: " << duration.count() << "s" << std::endl;
    Tensor2dXf ret_audio = CombineAudio(return_audio);
    return ret_audio;
}

Tensor3dXf Conv1D::forward(Tensor3dXf tensor, int InputChannels,
                           int OutputChannels, int kernel_size, int stride,
                           int padding)
{
    assert(conv_weights.dimension(0) == OutputChannels);
    assert(conv_weights.dimension(1) == InputChannels);
    assert(conv_weights.dimension(2) == kernel_size);
    assert(tensor.dimension(1) == InputChannels);

    int batch_size = tensor.dimension(0);
    int length = tensor.dimension(2);
    int padded_length = length + 2 * padding;
    int new_length = GetSize(padded_length, kernel_size, stride);
    Eigen::array<std::pair<int, int>, 3> paddings;
    paddings[0] = std::make_pair(0, 0);
    paddings[1] = std::make_pair(0, 0);
    paddings[2] = std::make_pair(padding, padding);

    Tensor3dXf padded_tensor = tensor.pad(paddings);
    tensor = padded_tensor;
    if (stride == 1 && kernel_size == 1) {
        Tensor4dXf newtensor = tensor.contract(
            conv_weights, Eigen::array<Eigen::IndexPair<int>, 1>{
                              {Eigen::IndexPair<int>(1, 1)}});
        Tensor3dXf ans =
            newtensor.shuffle(std::array<long long, 4>{3, 0, 2, 1}).chip(0, 0);
        Tensor3dXf bias_tensor =
            conv_bias.reshape(std::array<long long, 3>{1, OutputChannels, 1})
                .broadcast(std::array<long long, 3>{batch_size, 1, new_length});
        return ans + bias_tensor;
    }
    assert(kernel_size > 0 && kernel_size <= padded_length);
    assert(stride > 0 && stride <= padded_length);
    Tensor4dXf big_tensor(1, tensor.dimension(0), tensor.dimension(1),
                          tensor.dimension(2));
    big_tensor.chip(0, 0) = tensor;
    Tensor4dXf big_tensor_shuffled =
        big_tensor.shuffle(std::array<long long, 4>{1, 2, 3, 0});
    big_tensor = big_tensor_shuffled;
    Tensor5dXf extracted_images = big_tensor.extract_image_patches(
        InputChannels, kernel_size, InputChannels, stride, 1, 1,
        PaddingType::PADDING_VALID);
    Tensor5dXf extracted_images_shuffle =
        extracted_images.shuffle(std::array<long long, 5>{4, 0, 1, 2, 3});
    Tensor4dXf get_patches = extracted_images_shuffle.chip(0, 0);
    Tensor3dXf output_tensor = conv_weights.contract(
        get_patches,
        Eigen::array<Eigen::IndexPair<int>, 2>{
            {Eigen::IndexPair<int>(1, 1), Eigen::IndexPair<int>(2, 2)}});
    Tensor3dXf bias_tensor =
        conv_bias.reshape(std::array<long long, 3>{OutputChannels, 1, 1})
            .broadcast(std::array<long long, 3>{1, batch_size, new_length});
    output_tensor += bias_tensor;
    return output_tensor.shuffle(std::array<long long, 3>{1, 0, 2});
}