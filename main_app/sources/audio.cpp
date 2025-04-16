#include "audio.h"
#include "tensors.h"
#include <array>
#include <cassert>
#include <libnyquist/Common.h>
#include <libnyquist/Decoders.h>
#include <libnyquist/Encoders.h>
#include <stdexcept>
#include <unsupported/Eigen/CXX11/src/Tensor/Tensor.h>
using namespace nqr;
const int SUPPORTED_SAMPLE_RATE = 16000;
Tensor2dXf ReadAudioFromFile(fs::path file_name, int *sample_rate,
                             double *length_in_seconds)
{
    std::shared_ptr<AudioData> fileData = std::make_shared<AudioData>();

    NyquistIO loader;

    loader.Load(fileData.get(), file_name);
    // std::cout << "Input samples: "
    //           << fileData->samples.size() / fileData->channelCount <<
    //           std::endl;
    // std::cout << "Length in seconds: " << fileData->lengthSeconds <<
    // std::endl; std::cout << "Number of channels: " <<
    // fileData->channelCount<<std::endl<< "Sample Rate: "<<fileData->sampleRate
    // << std::endl;
    if (length_in_seconds) {
        *length_in_seconds = fileData->lengthSeconds;
    }
    if (sample_rate) {
        *sample_rate = fileData->sampleRate;
    }
    if (fileData->channelCount != 1) {
        throw std::invalid_argument(
            "demucs_mt.cpp only supports mono  audio");
    }
    int64_t N = fileData->samples.size() / fileData->channelCount;
    Tensor2dXf ret(1, N);
    if (fileData->channelCount == 1) {
        for (int64_t i = 0; i < N; ++i) {
            ret(0, i) = fileData->samples[i];
        }
    }
    if (fileData->sampleRate != SUPPORTED_SAMPLE_RATE) {
        ret = ResampleAudio(ret, fileData->sampleRate, SUPPORTED_SAMPLE_RATE);
    }
    return ret;
}
Tensor2dXf WriteAudioFromFile(fs::path file_name, Tensor2dXf wav,
                              int *sample_rate)
{
    fs::path directory = file_name.parent_path();
    if (!directory.empty() && !fs::exists(directory)) {
        fs::create_directories(directory);
    }
    std::shared_ptr<AudioData> fileData = std::make_shared<AudioData>();
    if (sample_rate && *sample_rate != SUPPORTED_SAMPLE_RATE) {
        fileData->sampleRate = *sample_rate;
        wav = ResampleAudio(wav, SUPPORTED_SAMPLE_RATE, *sample_rate);
    }
    else {
        fileData->sampleRate = SUPPORTED_SAMPLE_RATE;
    }

    fileData->channelCount = 1;
    fileData->samples.resize(wav.dimension(1));
    for (int64_t i = 0; i < wav.dimension(1); ++i) {
        fileData->samples[i] = wav(0, i);
    }
    int encoderStatus =
        encode_wav_to_disk({fileData->channelCount, PCM_FLT, DITHER_TRIANGLE},
                           fileData.get(), file_name);
    std::cout << "Encoder Status: " << encoderStatus << std::endl;
    return wav;
}

float compare_audio(Tensor2dXf noisy, Tensor2dXf clean)
{
    assert(noisy.dimensions() == clean.dimensions());
    Tensor2dXf difference = clean - noisy;
    Eigen::Tensor<float, 0> clean_sum =
        clean.unaryExpr([](float x) { return x * x; })
            .sum()
            .template cast<float>();
    float clean_power = clean_sum(0);
    Eigen::Tensor<float, 0> diff_sum =
        difference.unaryExpr([](float x) { return x * x; })
            .sum()
            .template cast<float>();
    float noise_power = diff_sum(0);
    float sdr = 10 * log10(clean_power / noise_power);
    return sdr;
}
float sinc(float f)
{
    return (f == 0.0f) ? 1.0f : (std::sin(f * M_PI) / (f * M_PI));
}

float hanning_func(int n, int N)
{
    return 0.5f * (1.0f - std::cos(2.0f * M_PI * float(n) / float(N - 1)));
}
Tensor1dXf compute_hanning_window(int64_t const p_window_size)
{
    Tensor1dXf window(p_window_size);
    for (int64_t i = 0; i < p_window_size; ++i)
        window[i] = hanning_func(i, p_window_size);
    return window;
}
Tensor1dXf compute_polyphase_filter_bank(int64_t upsampling_factor,
                                         int64_t downsampling_factor,
                                         int64_t filter_size)
{
    Tensor1dXf window = compute_hanning_window(filter_size * upsampling_factor);

    Tensor1dXf filter(window.size());
    float scale_factor =
        float(std::max(downsampling_factor, upsampling_factor));

    int64_t num_filters = upsampling_factor;
    int64_t num_filter_taps = window.size() / num_filters;

    for (int64_t i = 0; i < window.size(); ++i) {

        float t = int(i) - int(window.size() / 2);
        float f = sinc(t / scale_factor);

        int64_t polyphase_filter_index = i % num_filters;
        int64_t offset_in_polyphase_filter = i / num_filters;
        int64_t filter_array_idx =
            polyphase_filter_index * num_filter_taps +
            offset_in_polyphase_filter;
        filter[filter_array_idx] = f * window[i];
    }
    if (downsampling_factor > upsampling_factor) {
        filter =
            filter.unaryExpr([upsampling_factor, downsampling_factor](float x) {
                return (x * upsampling_factor) / downsampling_factor;
            });
    }
    return filter;
}

Tensor2dXf ResampleAudio(Tensor2dXf big_wav, int input_sample_rate,
                         int output_sample_rate)
{
    Tensor1dXf wav = big_wav.chip(0, 0);
    const int filter_size = 16;
    int64_t sample_rate_lcm = std::lcm(input_sample_rate, output_sample_rate);
    int64_t upsampling_factor = sample_rate_lcm / input_sample_rate;
    int64_t downsampling_factor = sample_rate_lcm / output_sample_rate;
    Tensor1dXf polyphase_filter_bank = compute_polyphase_filter_bank(
        upsampling_factor, downsampling_factor, filter_size);

    int64_t output_size =
        (wav.size() * upsampling_factor + downsampling_factor - 1) /
        downsampling_factor;
    Tensor2dXf output_wav(1, output_size);
    output_wav.setZero();
    int64_t num_polyphase_filters = upsampling_factor;
    long polyphase_filter_size =
        polyphase_filter_bank.size() / num_polyphase_filters;

    int64_t max_output_position =
        ((wav.size() - polyphase_filter_size) * upsampling_factor) /
        downsampling_factor;

    for (int64_t output_position = 0, interpolated_position = 0;
         output_position < max_output_position;
         output_position++, interpolated_position += downsampling_factor) {
        long polyphase_filter_index =
            (interpolated_position % num_polyphase_filters) *
            polyphase_filter_size;
        Tensor1dXf polyphase_filter_coefficients =
            polyphase_filter_bank
                .slice(std::array<long, 1>{polyphase_filter_index},
                       std::array<long, 1>{polyphase_filter_size})
                .reverse(Eigen::array<bool, 1>{true});

        long input_sample_position =
            static_cast<long>((output_position * downsampling_factor) /
                              static_cast<double>(upsampling_factor));

        Tensor1dXf input_samples_coefficients =
            wav.slice(std::array<long, 1>{input_sample_position},
                      std::array<long, 1>{polyphase_filter_size});
        Eigen::Tensor<float, 0> result =
            (input_samples_coefficients * polyphase_filter_coefficients)
                .sum()
                .template cast<float>();
        output_wav(0, output_position) = result(0);
    }
    return output_wav;
}