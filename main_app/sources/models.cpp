#include "coders.h"
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


void SimpleModel::forward(Tensor1dXf tensor)
{
    assert(tensor.dimension(0) == fc_weights.dimension(0));
    assert(fc_bias.dimension(0) == fc_weights.dimension(1));
    assert(tensor.size() > 0);
    result =
        (tensor.contract(fc_weights, product_dims_first_transposed)) + fc_bias;
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
