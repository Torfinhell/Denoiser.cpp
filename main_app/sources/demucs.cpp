#include "coders.h"
Tensor3dXf DemucsModel::EncoderWorker(Tensor3dXf mix, DemucsStreamer *streamer)
{
    Tensor3dXf mono, std;
    GetMean<Tensor3dXf, 3>(mix, mono, 1, indices_dim_3);
    if (streamer) {
        GetMean<Tensor3dXf, 3>(mono.unaryExpr([](float x) { return x * x; }),
                               std, 2, indices_dim_3);
        std_constant = std(std::array<long, 3>{0, 0, 0});
        assert(streamer->frames > 0 && "Frames should be at least one");
        streamer->variance =
            std_constant / static_cast<float>(streamer->frames) +
            (1 - 1 / static_cast<float>(streamer->frames)) * streamer->variance;
        assert(streamer->variance > 0 && "variance should be positive");
        mix = Tensor3dXf{mix.unaryExpr([streamer, this](float x) -> float {
            return x / (sqrt(streamer->variance) + floor);
        })};
        std::array<long, 3> offset = {0, 0,
                                      (streamer->stride) -
                                          (streamer->resample_buffer)},
                            extent = {1, 1, streamer->resample_buffer};
        Tensor3dXf padded_mix = streamer->resample_in.concatenate(mix, 2);
        streamer->resample_in = mix.slice(offset, extent);
        Tensor3dXf x = padded_mix;
        x = UpSample<Tensor3dXf, Tensor4dXf, 3>(x);
        x = UpSample<Tensor3dXf, Tensor4dXf, 3>(x);
        offset[2] = (streamer->resample) * (streamer->resample_buffer);
        extent[2] = (streamer->resample * streamer->frame_length);
        Tensor3dXf other = x.slice(offset, extent);
        x = other;
        skips.clear();
        streamer->next_state.clear();
        streamer->first = streamer->conv_state.empty();
        int frame_stride = streamer->stride * streamer->demucs_model.resample;
        Tensor3dXf prev;
        int demucs_stride = streamer->demucs_model.stride;
        int demucs_kernel_size = streamer->demucs_model.kernel_size;
        for (int i = 0; i < depth; i++) {
            frame_stride /= demucs_stride;
            length = x.dimension(2);
            if (i != depth - 1) {
                if (!streamer->first) {
                    prev = streamer->conv_state[0];
                    streamer->conv_state.erase(streamer->conv_state.begin());
                    extent = prev.dimensions();
                    offset[2] = frame_stride;
                    extent[2] = prev.dimension(2) - frame_stride;
                    prev = Tensor3dXf{prev.slice(offset, extent)};
                    int tgt = (length - demucs_kernel_size) / demucs_stride + 1;
                    int missing = tgt - prev.dimension(2);
                    int offset_x = length - demucs_kernel_size -
                                   demucs_stride * (missing - 1);
                    extent = x.dimensions();
                    offset[2] = offset_x;
                    extent[2] = x.dimension(2) - offset_x;
                    x = Tensor3dXf{x.slice(offset, extent)};
                }
                x = encoders[i].forward(x);
                if (!streamer->first) {
                    x = Tensor3dXf{prev.concatenate(x, 2)};
                }
                streamer->next_state.push_back(x);
                return streamer->next_state.back();
            }
            else {
                x = encoders[i].forward(x);
            }
            skips.push_back(x);
        }
        return x;
    }
    else {
        GetStd<Tensor3dXf, 3>(mono, std, 2, indices_dim_3);
        std_constant = std(std::array<long, 3>{0, 0, 0});
        assert((std_constant + floor) > 0 && "Must be positive");
        mix = mix.unaryExpr(
            [this](float x) { return x / (std_constant + floor); });
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
}


Tensor3dXf DemucsModel::DecoderWorker(Tensor3dXf mix, DemucsStreamer *streamer)
{
    std::array<long, 3> offset = {};
    std::array<long, 3> extent = {};
    Tensor3dXf x = mix;
    RELU relu;
    if (!streamer) {
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
    else {
        Tensor3dXf extra;
        Tensor3dXf prev;
        for (int i = 0; i < depth; i++) {
            int hidden = decoders[i].hidden;
            int ch_scale = decoders[i].ch_scale;
            int chout = decoders[i].chout;
            int kernel_size = decoders[i].kernel_size;
            int demucs_stride = decoders[i].stride;
            Tensor3dXf skip = skips[depth - 1 - i];
            offset[2]=0;
            extent = skip.dimensions();
            extent[2] = x.dimension(2);
            x = x + skip.slice(offset, extent);
            x = decoders[i].conv_1_1d.forward(x, hidden, hidden * ch_scale, 1,
                                              1);
            x = decoders[i].glu.forward(x);
            if (extra.size() > 0) {
                extent = skip.dimensions();
                offset[2] = x.dimension(2);
                extent[2] = skip.dimension(2) - offset[2];
                skip = Tensor3dXf{skip.slice(offset, extent)};
                extent = skip.dimensions();
                offset[2] = 0;
                extent[2] = extra.dimension(2);
                extra += skip.slice(offset, extent);
                extra = decoders[i].forward(extra);
            }
            x = decoders[i].conv_tr_1_1d.forward(x, hidden, chout, kernel_size,
                                                 demucs_stride);
            extent = x.dimensions();
            offset[2] = x.dimension(2) - demucs_stride;
            extent[2] = demucs_stride;
            Tensor3dXf colum_bias =
                decoders[i].conv_tr_1_1d.conv_tr_bias.reshape(
                    std::array<long, 3>{
                        1, decoders[i].conv_tr_1_1d.conv_tr_bias.size(), 1});
            streamer->next_state.push_back(
                x.slice(offset, extent) -
                colum_bias.broadcast(
                    std::array<long, 3>{x.dimension(0), 1, demucs_stride}));
            Eigen::array<std::pair<int, int>, 3> paddings = {};
            if (extra.size() == 0) {
                extra = x.slice(offset, extent);
            }
            else {
                paddings[2] = {0, extra.dimension(2) -
                                      demucs_stride};
                extra += streamer->next_state.back().pad(paddings);
            }
            offset[2] = 0;
            extent[2] = x.dimension(2) - demucs_stride;
            x = Tensor3dXf{x.slice(offset, extent)};
            if (!streamer->first) {
                prev = streamer->conv_state[0];
                streamer->conv_state.erase(streamer->conv_state.begin());
                paddings[2] = {0, x.dimension(2) - demucs_stride};
                x += prev.pad(paddings);
            }
            if (i != depth - 1) {
                x = relu.forward(x);
                extra = relu.forward(extra);
            }
        }
        Tensor3dXf out = x;
        streamer->conv_state = streamer->next_state;
        Tensor3dXf padded_out =
            Tensor3dXf{streamer->resample_out.concatenate(out, 2)}.concatenate(
                extra, 2);
        extent = out.dimensions();
        offset[2] = out.dimension(2) - streamer->resample_buffer;
        extent[2] = streamer->resample_buffer;
        streamer->resample_out = out.slice(offset, extent);
        out = DownSample<Tensor3dXf, Tensor4dXf, 3>(padded_out);
        out = DownSample<Tensor3dXf, Tensor4dXf, 3>(out);
        extent = out.dimensions();
        offset[2] = (streamer->resample_buffer) / (streamer->resample);
        extent[2] = streamer->stride;
        out = Tensor3dXf{out.slice(offset, extent)};
        out = Tensor3dXf{out.unaryExpr([this, streamer](float x) {
            return sqrt(streamer->variance) * x;
        })};
        
        return out;
    }
}

Tensor3dXf DemucsModel::LSTMWorker(Tensor3dXf mix, LstmState &lstm_state)
{
    auto res1 = lstm1.forward(mix.shuffle(std::array<long long, 3>{2, 0, 1}),
                              lstm_hidden, lstm_state);
    lstm_state.is_created = false;
    auto res2 = lstm2.forward(res1, lstm_hidden, lstm_state);
    auto res3=res2.shuffle(std::array<long long, 3>{1, 2, 0});
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

Tensor3dXf DemucsModel::forward(Tensor3dXf mix, LstmState &lstm_state,
    DemucsStreamer *streamer)
{
// auto start = std::chrono::high_resolution_clock::now();
auto res1 = EncoderWorker(mix, streamer);
// auto end = std::chrono::high_resolution_clock::now();
// std::cout << "Time taken for EncoderWorker: "
//           << std::chrono::duration<double>(end - start).count()
//           << " seconds" << std::endl;
// start = std::chrono::high_resolution_clock::now();
auto res2 = LSTMWorker(res1, lstm_state);
// end = std::chrono::high_resolution_clock::now();
// std::cout << "Time taken for LSTMWorker: "
//           << std::chrono::duration<double>(end - start).count()
//           << " seconds" << std::endl;
// start = std::chrono::high_resolution_clock::now();
auto res3 = DecoderWorker(res2, streamer);
// end = std::chrono::high_resolution_clock::now();
// std::cout << "Time taken for DecoderWorker: "
//           << std::chrono::duration<double>(end - start).count()
//           << " seconds" << std::endl;
return res3;//////////////////////////////////////
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