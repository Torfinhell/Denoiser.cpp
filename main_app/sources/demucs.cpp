#include "coders.h"
#include "layers.h"
Tensor3dXf DemucsModel::EncoderWorker(Tensor3dXf mix)
{
    Tensor3dXf mono, std;
    GetMean<Tensor3dXf, 3>(mix, mono, 1, indices_dim_3);
    GetStd<Tensor3dXf, 3>(mono, std, 2, indices_dim_3);
    std_constant = std(std::array<long, 3>{0, 0, 0});
    assert((std_constant + floor) > 0 && "Must be positive");
    mix = mix.unaryExpr([this](float x) { return x / (std_constant + floor); });
    length = mix.dimension(mix.NumDimensions - 1);
    Tensor3dXf x = mix;
    Eigen::array<std::pair<int, int>, 3> paddings = {};
    paddings[2] = std::make_pair(0, valid_length(length) - length);
    Tensor3dXf padded_x = x.pad(paddings);
    x = padded_x;
    x = UpSample<Tensor3dXf, Tensor4dXf, 3>(x);
    x = UpSample<Tensor3dXf, Tensor4dXf, 3>(x);
    skips.clear();
    for (int i = 0; i < depth; i++) {
        x = encoders[i].forward(x);
        skips.push_back(x);
    }
    return x;
}

Tensor3dXf DemucsModel::DecoderWorker(Tensor3dXf mix)
{
    std::array<long, 3> offset = {};
    std::array<long, 3> extent = {};
    Tensor3dXf x = mix;
    RELU relu;
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
    x = DownSample<Tensor3dXf, Tensor4dXf, 3>(x);
    x = DownSample<Tensor3dXf, Tensor4dXf, 3>(x);
    extent = x.dimensions();
    extent[2] = length;
    Tensor3dXf x_slice = x.slice(offset, extent);
    return x_slice.unaryExpr([this](float x) { return std_constant * x; });
}

Tensor3dXf DemucsModel::LSTMWorker(Tensor3dXf mix)
{
    auto res1 = lstm1.forward(mix.shuffle(std::array<long long, 3>{2, 0, 1}),
                              lstm_hidden, lstm_state1);
    auto res2 = lstm2.forward(res1, lstm_hidden, lstm_state2);
    auto res3 = res2.shuffle(std::array<long long, 3>{1, 2, 0});
    return res3;
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
    assert(resample > 0 && "resample should be positive");
    length = static_cast<int>(std::ceil(length / resample));
    return length;
}

Tensor3dXf DemucsModel::forward(Tensor3dXf mix)
{
    auto res1 = EncoderWorker(mix);
    auto res2 = LSTMWorker(res1);
    auto res3 = DecoderWorker(res2);
    return res3;
}

bool DemucsModel::load_from_jit_module(const torch::jit::script::Module &module)
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

float DemucsModel::MaxAbsDifference(const DemucsModel &other) const
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

bool DemucsModel::IsEqual(const DemucsModel &other, float tolerance) const
{
    return MaxAbsDifference(other) <= tolerance;
}

DemucsModel::DemucsModel(int hidden, int ch_scale, int kernel_size, int stride,
                         int chout, int depth, int chin, int max_hidden,
                         int growth, int resample, float floor)
    : hidden(hidden), ch_scale(ch_scale), kernel_size(kernel_size),
      stride(stride), chout(chout), depth(depth), resample(resample),
      chin(chin), floor(floor)
{
    int chin_first = chin;
    int chout_first = chout;
    int hidden_first = hidden;
    for (int i = 0; i < depth; i++) {
        encoders.emplace_back(hidden, ch_scale, kernel_size, stride, chout,
                              chin);
        decoders.emplace_back(hidden, ch_scale, kernel_size, stride, chout);
        chout = hidden;
        chin = hidden;
        hidden = std::min(int(growth * hidden), max_hidden);
    }
    std::reverse(decoders.begin(), decoders.end());
    lstm_hidden = chin;
    hidden = hidden_first;
    chin = chin_first;
    chout = chout_first;
}
