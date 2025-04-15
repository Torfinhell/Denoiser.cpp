#include "coders.h"
#include "layers.h"
Eigen::array<Eigen::IndexPair<int>, 1> product_dims_reg = {
    Eigen::IndexPair<int>(1, 0)};
Eigen::array<Eigen::IndexPair<int>, 1> product_dims_sec_transposed = {
    Eigen::IndexPair<int>(1, 1)};
Eigen::array<Eigen::IndexPair<int>, 1> product_dims_first_transposed = {
    Eigen::IndexPair<int>(0, 0)};
Eigen::array<Eigen::IndexPair<int>, 1> product_dims_both_transposed = {
    Eigen::IndexPair<int>(0, 1)};
using namespace Tensors;
Tensor3dXf KernelUpsample2(int zeros)
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
    auto sinc_func = [](float x) { return std::sin(x) / x; };
    Tensor1dXf kernel = (t.unaryExpr(sinc_func)) * window_odd;
    Tensor3dXf kernel_reshaped =
        kernel.reshape(std::array<long, 3>{1, 1, kernel.size()});
    return kernel_reshaped;
}

Tensor3dXf OneEncoder::forward(Tensor3dXf tensor)
{
    auto res1 = conv_1_1d.forward(tensor, chin, hidden, kernel_size, stride);
    auto res2 = relu.forward(res1);
    auto res3 = conv_2_1d.forward(res2, hidden, hidden * ch_scale, 1);
    auto res4 = glu.forward(res3);
    return res4;
}

bool OneEncoder::load_from_jit_module(const torch::jit::script::Module &module,
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
        std::cerr << "Other exception: " << e.what() << std::endl;
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

float OneEncoder::MaxAbsDifference(const OneEncoder &other) const 
{
    return max_of_multiple({conv_1_1d.MaxAbsDifference(other.conv_1_1d),
                            conv_2_1d.MaxAbsDifference(other.conv_2_1d)});
}

bool OneEncoder::IsEqual(const OneEncoder &other, float tolerance) const
{
    return MaxAbsDifference(other) <= tolerance;
}
OneEncoder::OneEncoder(int hidden, int ch_scale, int kernel_size, int stride,
                       int chout, int chin)
    : hidden(hidden), ch_scale(ch_scale), kernel_size(kernel_size),
      stride(stride), chout(chout), chin(chin)
{
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
    assert(kernel_size > 0 && kernel_size <= padded_length &&
           "Invalid kernel size");
    assert(stride > 0 && stride <= padded_length);
    Tensor4dXf big_tensor(1, tensor.dimension(0), tensor.dimension(1),
                          tensor.dimension(2));
    big_tensor.chip(0, 0) = tensor;
    Tensor4dXf big_tensor_shuffled =
        big_tensor.shuffle(std::array<long long, 4>{1, 2, 3, 0});
    big_tensor = big_tensor_shuffled;
    Tensor5dXf extracted_images = big_tensor.extract_image_patches(
        InputChannels, kernel_size, InputChannels, stride, 1, 1,
        Eigen::PaddingType::PADDING_VALID);
    Tensor5dXf extracted_images_shuffle =
        extracted_images.shuffle(std::array<long long, 5>{4, 0, 1, 2, 3});
    Tensor4dXf get_patches = extracted_images_shuffle.chip(0, 0);
    Tensor3dXf output_tensor(OutputChannels, batch_size, new_length);
    output_tensor = conv_weights.contract(
        get_patches,
        Eigen::array<Eigen::IndexPair<int>, 2>{
            {Eigen::IndexPair<int>(1, 1), Eigen::IndexPair<int>(2, 2)}});
    Tensor3dXf bias_tensor =
        conv_bias.reshape(std::array<long long, 3>{OutputChannels, 1, 1})
            .broadcast(std::array<long long, 3>{1, batch_size, new_length});
    output_tensor += bias_tensor;
    return output_tensor.shuffle(std::array<long long, 3>{1, 0, 2});
}
bool Conv1D::load_from_jit_module(const torch::jit::script::Module &module,
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
        std::cerr << "Other exception: " << e.what() << std::endl;
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

float Conv1D::MaxAbsDifference(const Conv1D &other) const 
{
    return max_of_multiple(
        {::MaxAbsDifference(conv_bias, other.conv_bias), 
         ::MaxAbsDifference(conv_weights, other.conv_weights)}); 
}

bool Conv1D::IsEqual(const Conv1D &other, float tolerance) const
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
    Tensor3dXf A = tensor.slice(offset1, extent);
    Tensor3dXf B = tensor.slice(offset2, extent);
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

bool OneDecoder::load_from_jit_module(const torch::jit::script::Module &module,
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
        std::cerr << "Other exception: " << e.what() << std::endl;
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

float OneDecoder::MaxAbsDifference(const OneDecoder &other) const 
{
    return max_of_multiple<float>(
        {conv_1_1d.MaxAbsDifference(other.conv_1_1d),
         conv_tr_1_1d.MaxAbsDifference(other.conv_tr_1_1d)});
}

bool OneDecoder::IsEqual(const OneDecoder &other, float tolerance) const
{
    return MaxAbsDifference(other) <= tolerance;
}

OneDecoder::OneDecoder(int hidden, int ch_scale, int kernel_size, int stride,
                       int chout)
    : hidden(hidden), ch_scale(ch_scale), kernel_size(kernel_size),
      stride(stride), chout(chout)
{
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
    // we assume that kernel_size==stride*2
    auto func_get_ans =
        [batch_size, OutputChannels, length,
         stride](const Tensor3dXf &input, const Tensor3dXf &stride_kernel,
                 int padding_left = 0, int padding_right = 0) -> Tensor3dXf {
        Eigen::array<std::pair<int, int>, 3> paddings;
        paddings[0] = std::make_pair(0, 0);
        paddings[1] = std::make_pair(0, 0);
        paddings[2] = std::make_pair(padding_left, padding_right);
        int new_length_for_stride = stride * length;
        Tensor4dXf newtensor = input.contract(
            stride_kernel, Eigen::array<Eigen::IndexPair<int>, 1>{
                               {Eigen::IndexPair<int>(1, 0)}});
        Tensor4dXf other_tensor =
            newtensor.shuffle(std::array<long long, 4>{0, 2, 3, 1});
        Tensor3dXf res = other_tensor.reshape(std::array<long, 3>{
            batch_size, OutputChannels, new_length_for_stride});
        return res.pad(paddings);
    };
    std::array<long, 3> offset = {0, 0, 0};
    std::array<long, 3> offset1 = {0, 0, stride};
    std::array<long, 3> extent = {InputChannels, OutputChannels, stride};
    Tensor3dXf output_tensor =
        func_get_ans(tensor, conv_tr_weights.slice(offset, extent), 0, stride) +
        func_get_ans(tensor, conv_tr_weights.slice(offset1, extent), stride, 0);
    output_tensor +=
        conv_tr_bias.reshape(std::array<long, 3>{1, OutputChannels, 1})
            .broadcast(std::array<long long, 3>{batch_size, 1, new_length});
    return output_tensor;
}

bool ConvTranspose1d::load_from_jit_module(
    const torch::jit::script::Module &module, std::string weights_index)
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
        std::cerr << "Other exception: " << e.what() << std::endl;
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

float ConvTranspose1d::MaxAbsDifference(const ConvTranspose1d &other) const 
{
    return max_of_multiple(
        {::MaxAbsDifference(conv_tr_bias, other.conv_tr_bias), 
         ::MaxAbsDifference(conv_tr_weights, other.conv_tr_weights)}); 
}

bool ConvTranspose1d::IsEqual(const ConvTranspose1d &other, float tolerance) const
{
    return MaxAbsDifference(other) <= tolerance;
}

int ConvTranspose1d::GetTransposedSize(int size, int kernel_size, int stride)
{
    return stride * (size - 1) + kernel_size;
}

Tensor3dXf OneLSTM::forward(Tensor3dXf tensor, int HiddenSize,
                            LstmState &lstm_state)
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
    // assert(lstm_weight_hh.dimension(0) == 4 * HiddenSize &&
    //        lstm_weight_hh.dimension(1) == HiddenSize);
    // assert(lstm_weight_ih.dimension(0) == 4 * HiddenSize &&
    //        lstm_weight_ih.dimension(1) == input_size);
    // assert(lstm_bias_ih.dimension(0) == 4 * HiddenSize &&
    //        lstm_bias_ih.dimension(1) == 1);
    // assert(lstm_bias_hh.dimension(0) == 4 * HiddenSize &&
    //        lstm_bias_hh.dimension(1) == 1);
    for (int t = 0; t < length; t++) {
        Tensor2dXf x_t = tensor.chip(t, 0); 
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

bool OneLSTM::load_from_jit_module(const torch::jit::script::Module &module,
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
        std::cerr << "Other exception: " << e.what() << std::endl;
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

float OneLSTM::MaxAbsDifference(const OneLSTM &other)  const
{
    return max_of_multiple<float>({
        ::MaxAbsDifference(lstm_weight_ih, other.lstm_weight_ih),  
        ::MaxAbsDifference(lstm_weight_hh, other.lstm_weight_hh),  
        ::MaxAbsDifference(lstm_bias_ih, other.lstm_bias_ih),  
        ::MaxAbsDifference(lstm_bias_hh, other.lstm_bias_hh),  
    });
}

bool OneLSTM::IsEqual(const OneLSTM &other, float tolerance) const
{
    return MaxAbsDifference(other) <= tolerance;
}
