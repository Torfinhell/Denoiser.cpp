#include "audio.h"
#include "layers.h"
#include <Eigen/src/Core/Matrix.h>
#include <array>
#include <cassert>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>
DemucsStreamer::DemucsStreamer(int stride, int numThreads)
    : stride(stride), numThreads(numThreads)
{
    assert(stride>0 && "Stride should be positive");
    assert(numThreads>0 && "Number of Threads should be positive");
    demucs_models.resize(numThreads);
    fs::path model_path = "../tests/test_data/dns48";
    std::ifstream input_file;
    input_file.open(model_path / "data.txt");
    if (!input_file) {
        throw std::runtime_error("Unable to open model file.");
    }
    demucs_models[0].load_from_file(input_file);
    input_file.close();
    for (int i = 1; i < numThreads; i++) {
        demucs_models[i] = demucs_models[i - 1];
    }
}
Tensor2dXf DemucsStreamer::forward(Tensor2dXf wav)
{
    std::vector<std::thread> threads_encoders;
    std::vector<std::thread> threads_decoders;
    Tensor3dXf big_wav(1, wav.dimension(0), wav.dimension(1));
    big_wav.chip(0, 0) = wav;
    std::vector<Tensor3dXf> buffer_audio;
    long long frame_size = stride;
    long long counter=0;
    for (long long i = 0; i < big_wav.dimension(2); i += frame_size) {
        std::array<long, 3> offset = {0, 0, i};
        std::array<long, 3> extent = {1, big_wav.dimension(1),
                                      (frame_size < big_wav.dimension(2) - i)
                                          ? frame_size
                                          : big_wav.dimension(2) - i};
        buffer_audio.push_back(big_wav.slice(offset, extent));
    }
    std::vector<Tensor3dXf> encoders_result(buffer_audio.size());
    std::vector<Tensor3dXf> lstm_result(buffer_audio.size());
    std::vector<Tensor3dXf> decoders_result(buffer_audio.size());
    for (int start_thread = 0; start_thread < buffer_audio.size();
            start_thread += numThreads) {
        for (int i = start_thread;
                i < std::min(static_cast<int>(buffer_audio.size()),
                            start_thread + numThreads);
                i++) {
            threads_encoders.emplace_back([&, i]() {
                encoders_result[i] =
                    demucs_models[i % numThreads].EncoderWorker(
                        buffer_audio[i]);
            });
        }
        for (auto &t : threads_encoders) {
            t.join();
        }
        for (int i = start_thread;
                i < std::min(static_cast<int>(buffer_audio.size()),
                            start_thread + numThreads);
                i++) {
            lstm_result[i] = demucs_models[i % numThreads].LSTMWorker(
                encoders_result[i]);
                demucs_models[(i + 1) % numThreads].lstm_state1 =
                demucs_models[i % numThreads].lstm_state1;
                demucs_models[(i + 1) % numThreads].lstm_state2 =
                demucs_models[i % numThreads].lstm_state2;
        }
        for (int i = start_thread;
            i < std::min(static_cast<int>(buffer_audio.size()),
                         start_thread + numThreads);
            i++) {
           threads_decoders.emplace_back([&, i]() {
               decoders_result[i] =
                   demucs_models[i % numThreads].DecoderWorker(lstm_result[i]);
           });
       }
       for (auto &t : threads_decoders) {
           t.join();
       }
       threads_decoders.clear();
       threads_encoders.clear();
    }
    Eigen::MatrixXf ret_audio;
    for (int i = 0; i < decoders_result.size(); i++) {
        int height = decoders_result[i].dimension(1);
        int width = decoders_result[i].dimension(2);

        Eigen::MatrixXf matrix = Eigen::Map<Eigen::MatrixXf>(
            decoders_result[i].data(), height, width);
        if (i == 0) {
            ret_audio = matrix;
        }
        else {
            ret_audio.conservativeResize(Eigen::NoChange,
                                         ret_audio.cols() + matrix.cols());
            ret_audio.rightCols(matrix.cols()) = matrix;
        }
    }
    Tensor2dXf final_tensor = Eigen::TensorMap<Tensor2dXf>(
        ret_audio.data(), ret_audio.rows(), ret_audio.cols());
    return final_tensor;
}
