#include "audio.h"
#include "tensors.h"
#include <libnyquist/Common.h>
#include <libnyquist/Decoders.h>
#include <libnyquist/Encoders.h>
#include <stdexcept>
#include <unsupported/Eigen/CXX11/src/Tensor/Tensor.h>
using namespace nqr;
const int SUPPORTED_SAMPLE_RATE = 16000;
Tensor2dXf ReadAudioFromFile(fs::path file_name, double *length_in_seconds)
{
    std::shared_ptr<AudioData> fileData = std::make_shared<AudioData>();

    NyquistIO loader;

    loader.Load(fileData.get(), file_name);
    // if (fileData->sampleRate != SUPPORTED_SAMPLE_RATE) {
    //     std::cerr << "[ERROR] demucs_mt.cpp only supports the following sample "
    //                  "rate (Hz): "
    //               << SUPPORTED_SAMPLE_RATE << std::endl;
    //     exit(1);
    // }
    // std::cout << "Input samples: "
    //           << fileData->samples.size() / fileData->channelCount <<
    //           std::endl;
    // std::cout << "Length in seconds: " << fileData->lengthSeconds <<
    // std::endl; std::cout << "Number of channels: " << fileData->channelCount
    // << std::endl;
    if(length_in_seconds){
        *length_in_seconds=fileData->lengthSeconds;
    }
    if (fileData->channelCount != 1) {
        throw std::runtime_error(
            "[ERROR] demucs_mt.cpp only supports mono  audio");
    }
    size_t N = fileData->samples.size() / fileData->channelCount;
    Tensor2dXf ret(1, N);
    if (fileData->channelCount == 1) {
        for (size_t i = 0; i < N; ++i) {
            ret(0, i) = fileData->samples[i];
        }
    }
    return ret;
}
void WriteAudioFromFile(fs::path file_name, Tensor2dXf wav)
{
    fs::path directory = file_name.parent_path();
    if (!directory.empty() && !fs::exists(directory)) {
        fs::create_directories(directory);
    }
    std::shared_ptr<AudioData> fileData = std::make_shared<AudioData>();
    fileData->sampleRate = SUPPORTED_SAMPLE_RATE;
    fileData->channelCount = 1;
    fileData->samples.resize(wav.dimension(1));
    for (long int i = 0; i < wav.dimension(1); ++i) {
        fileData->samples[i] = wav(0, i);
    }
    int encoderStatus =
        encode_wav_to_disk({fileData->channelCount, PCM_FLT, DITHER_TRIANGLE},
                           fileData.get(), file_name);
    std::cout << "Encoder Status: " << encoderStatus << std::endl;
}


float compare_audio(Tensor2dXf noisy, Tensor2dXf clean) {
    std::cout<<noisy.dimensions()<<clean.dimensions()<<std::endl;
    Tensor2dXf difference = clean - noisy;
    Eigen::Tensor<float, 0> clean_sum= clean.unaryExpr([](float x) { return x * x; }).sum().template cast<float>(); // Use cast to get the scalar value

    float clean_power =std::sqrt(clean_sum(0));
    Eigen::Tensor<float, 0> diff_sum = difference.unaryExpr([](float x) { return x * x; }).sum().template cast<float>(); // Us
    float noise_power =std::sqrt(clean_sum(0));
    float sdr = 10 * log10(clean_power / noise_power);
    return sdr;
}