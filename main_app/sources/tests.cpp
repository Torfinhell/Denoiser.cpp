#include "tests.h"
#include "audio.h"
#include "layers.h"
#include "tensors.h"
#include <exception>
#include <filesystem>
#include <ios>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
std::array<long, 1> indices_dim_1 = {0};
std::array<long, 2> indices_dim_2 = {0, 0};
std::array<long, 3> indices_dim_3 = {0, 0, 0};
std::array<long, 4> indices_dim_4 = {0, 0, 0, 0};
namespace fs = std::filesystem;
using namespace Tensors;
void print_model_layers(const torch::jit::script::Module &module)
{
    for (const auto &child : module.named_children()) {
        std::cout << "Layer: " << child.name
                  << ", Type: " << child.value.type()->str() << std::endl;
        for (const auto &param : child.value.named_parameters()) {
            std::cout << "  Param: " << param.name
                      << ", Shape: " << param.value.sizes() << std::endl;
        }
        print_model_layers(child.value);
    }
}

void TestSimpleModel()
{
    try {
        fs::path base_path = "../tests/test_data/SimpleModel";
        fs::path input_path = base_path / "input.pth";
        fs::path model_path = base_path / "model.pth";
        fs::path prediction_path = base_path / "prediction.pth";
        assert(fs::exists(input_path) && "Input path does not exist");
        assert(fs::exists(model_path) && "Model path does not exist");
        assert(fs::exists(prediction_path) && "Prediction path does not exist");
        torch::jit::script::Module prior_input, model, prior_prediction;
        torch::Tensor input_tensors, prediction_tensors;

        try {
            model = torch::jit::load(model_path);
            prior_input = torch::jit::load(input_path);
            prior_prediction = torch::jit::load(prediction_path);
            input_tensors = prior_input.attr("prior").toTensor();
            prediction_tensors = prior_prediction.attr("prior").toTensor();
            std::cout << "Simple Model loaded successfully" << std::endl;
        }
        catch (const c10::Error &e) {
            throw std::runtime_error("Error loading the model: " +
                                     std::string(e.what()));
        }
        assert(input_tensors.dim() == 1 && "Input tensor must be 1D");
        assert(prediction_tensors.dim() == 1 && "Prediction tensor must be 1D");
        Tensor1dXf input = TorchToEigen<Tensor1dXf, 1>(input_tensors);
        Tensor1dXf prediction = TorchToEigen<Tensor1dXf, 1>(prediction_tensors.detach());
        SimpleModel simple_model, loaded_model;

        if (!simple_model.load_from_jit_module(model)) {
            throw std::runtime_error("Couldn't load model from jit.\n");
        }

        std::ios::sync_with_stdio(false);
        std::ofstream output_file(base_path / "data.txt");
        if (!output_file) {
            throw std::runtime_error("Error opening data.txt file!");
        }
        simple_model.load_to_file(output_file);
        if (!output_file.good()) {
            throw std::runtime_error("Error writing to file!");
        }
        output_file.close();

        std::ifstream input_file(base_path / "data.txt");
        if (!input_file) {
            throw std::runtime_error("Error opening data.txt file!");
        }
        loaded_model.load_from_file(input_file);
        input_file.close();

        if (!simple_model.IsEqual(loaded_model, 1e-5)) {
            throw std::runtime_error(
                "Model is not correctly loaded. The Max difference between "
                "their "
                "elements: " +
                std::to_string(simple_model.MaxAbsDifference(loaded_model)));
        }

        simple_model.forward(input);
        if (!TestIfEqual<Tensor1dXf>(prediction, simple_model.result)) {
            throw std::runtime_error(
                "Error: Comparison of our prediction and known output failed");
        }
    }
    catch (const std::exception &e) {
        std::cout << "Simple Model Test not passed: \n"
                  << "Error occurred in file: " << __FILE__
                  << " at line: " << __LINE__ << "\n"
                  << e.what() << std::endl;
        return;
    }
    std::cout << "Simple Model Test Successfully passed" << std::endl;
}
void TestOneEncoder()
{
    try {
        fs::path base_path = "../tests/test_data/OneEncoder";
        fs::path input_path = base_path / "input.pth";
        fs::path model_path = base_path / "model.pth";
        fs::path prediction_path = base_path / "prediction.pth";
        assert(fs::exists(input_path) && "Input path does not exist");
        assert(fs::exists(model_path) && "Model path does not exist");
        assert(fs::exists(prediction_path) && "Prediction path does not exist");
        torch::jit::script::Module prior_input, model, prior_prediction;
        torch::Tensor input_tensors, prediction_tensors;

        try {
            model = torch::jit::load(model_path);
            prior_input = torch::jit::load(input_path);
            prior_prediction = torch::jit::load(prediction_path);
            input_tensors = prior_input.attr("prior").toTensor();
            prediction_tensors = prior_prediction.attr("prior").toTensor();
            std::cout << "Encoder Model loaded successfully" << std::endl;
        }
        catch (const c10::Error &e) {
            throw std::runtime_error("Error loading the model: " +
                                     std::string(e.what()));
        }
        assert(input_tensors.dim() == 3 && "Input tensor must be 3D");
        assert(prediction_tensors.dim() == 3 && "Prediction tensor must be 3D");
        Tensor3dXf input = TorchToEigen<Tensor3dXf, 3>(input_tensors);
        Tensor3dXf prediction = TorchToEigen<Tensor3dXf, 3>(prediction_tensors.detach());
        OneEncoder one_encoder_model, loaded_model;
        if (!one_encoder_model.load_from_jit_module(model)) {
            throw std::runtime_error("Couldn't load model from jit.\n");
        }
        std::ios::sync_with_stdio(false);
        std::ofstream output_file(base_path / "data.txt");
        if (!output_file) {
            throw std::runtime_error("Error opening data.txt file!");
        }
        one_encoder_model.load_to_file(output_file);
        if (!output_file.good()) {
            throw std::runtime_error("Error writing to file!");
        }
        output_file.close();

        std::ifstream input_file(base_path / "data.txt");
        if (!input_file) {
            throw std::runtime_error("Error opening data.txt file!");
        }
        loaded_model.load_from_file(input_file);
        input_file.close();

        if (!one_encoder_model.IsEqual(loaded_model, 1e-5)) {
            throw std::runtime_error(
                "Model is not correctly loaded. The Max difference between "
                "their "
                "elements: " +
                std::to_string(
                    one_encoder_model.MaxAbsDifference(loaded_model)));
        }
        if (!TestIfEqual<Tensor3dXf>(prediction,
                                     one_encoder_model.forward(input))) {
            throw std::runtime_error(
                "Error: Comparison of our prediction and known output failed."
                "The Absolute difference is: " +
                std::to_string(MaxAbsDifference(
                    prediction, one_encoder_model.forward(input))));
        }
    }
    catch (const std::exception &e) {
        std::cout << "OneEncoder Model Test not passed: \n"
                  << "Error occurred in file: " << __FILE__
                  << " at line: " << __LINE__ << "\n"
                  << e.what() << std::endl;
        return;
    }
    std::cout << "OneEncoder Model Test Successfully passed" << std::endl;
}

void TestSimpleEncoderDecoder()
{
    try {
        fs::path base_path = "../tests/test_data/SimpleEncoderDecoder";
        fs::path input_path = base_path / "input.pth";
        fs::path model_path = base_path / "model.pth";
        fs::path prediction_path = base_path / "prediction.pth";
        assert(fs::exists(input_path) && "Input path does not exist");
        assert(fs::exists(model_path) && "Model path does not exist");
        assert(fs::exists(prediction_path) && "Prediction path does not exist");
        torch::jit::script::Module prior_input, model, prior_prediction;
        torch::Tensor input_tensors, prediction_tensors;
        try {
            model = torch::jit::load(model_path);
            prior_input = torch::jit::load(input_path);
            prior_prediction = torch::jit::load(prediction_path);
            input_tensors = prior_input.attr("prior").toTensor();
            prediction_tensors = prior_prediction.attr("prior").toTensor();
            std::cout << "SimpleEncoderDecoder Model loaded successfully"
                      << std::endl;
        }
        catch (const c10::Error &e) {
            throw std::runtime_error("Error loading the model: " +
                                     std::string(e.what()));
        }
        assert(input_tensors.dim() == 3 && "Input tensor must be 3D");
        assert(prediction_tensors.dim() == 3 && "Prediction tensor must be 3D");
        Tensor3dXf input = TorchToEigen<Tensor3dXf, 3>(input_tensors);
        Tensor3dXf prediction = TorchToEigen<Tensor3dXf, 3>(prediction_tensors.detach());
        SimpleEncoderDecoder simple_encoder_decoder_model, loaded_model;
        if (!simple_encoder_decoder_model.load_from_jit_module(model)) {
            throw std::runtime_error("Couldn't load model from jit.\n");
        }
        std::ios::sync_with_stdio(false);
        std::ofstream output_file(base_path / "data.txt");
        if (!output_file) {
            throw std::runtime_error("Error opening data.txt file!");
        }
        simple_encoder_decoder_model.load_to_file(output_file);
        if (!output_file.good()) {
            throw std::runtime_error("Error writing to file!");
        }
        output_file.close();

        std::ifstream input_file(base_path / "data.txt");
        if (!input_file) {
            throw std::runtime_error("Error opening data.txt file!");
        }
        loaded_model.load_from_file(input_file);
        input_file.close();

        if (!simple_encoder_decoder_model.IsEqual(loaded_model, 1e-5)) {
            throw std::runtime_error(
                "Model is not correctly loaded. The Max difference between "
                "their "
                "elements: " +
                std::to_string(simple_encoder_decoder_model.MaxAbsDifference(
                    loaded_model)));
        }
        if (!TestIfEqual<Tensor3dXf>(
                prediction, simple_encoder_decoder_model.forward(input))) {
            throw std::runtime_error(
                "Error: Comparison of our prediction and known output failed."
                "The Absolute difference is: " +
                std::to_string(MaxAbsDifference(
                    prediction, simple_encoder_decoder_model.forward(input))));
        }
    }
    catch (const std::exception &e) {
        std::cout << "SimpleEncoderDecoder Model Test not passed: \n"
                  << "Error occurred in file: " << __FILE__
                  << " at line: " << __LINE__ << "\n"
                  << e.what() << std::endl;
        return;
    }
    std::cout << "SimpleEncoderDecoder Model Test Successfully passed"
              << std::endl;
}

void TestSimpleEncoderDecoderLSTM()
{
    try {
        fs::path base_path = "../tests/test_data/SimpleEncoderDecoderLSTM";
        fs::path input_path = base_path / "input.pth";
        fs::path model_path = base_path / "model.pth";
        fs::path prediction_path = base_path / "prediction.pth";
        assert(fs::exists(input_path) && "Input path does not exist");
        assert(fs::exists(model_path) && "Model path does not exist");
        assert(fs::exists(prediction_path) && "Prediction path does not exist");
        torch::jit::script::Module prior_input, model, prior_prediction;
        torch::Tensor input_tensors, prediction_tensors;

        try {
            model = torch::jit::load(model_path);
            prior_input = torch::jit::load(input_path);
            prior_prediction = torch::jit::load(prediction_path);
            input_tensors = prior_input.attr("prior").toTensor();
            prediction_tensors = prior_prediction.attr("prior").toTensor();
            std::cout << "SimpleEncoderDecoderLSTM Model loaded successfully"
                      << std::endl;
        }
        catch (const c10::Error &e) {
            throw std::runtime_error("Error loading the model: " +
                                     std::string(e.what()));
        }
        assert(input_tensors.dim() == 3 && "Input tensor must be 3D");
        assert(prediction_tensors.dim() == 3 && "Prediction tensor must be 3D");
        Tensor3dXf input = TorchToEigen<Tensor3dXf, 3>(input_tensors);
        Tensor3dXf prediction = TorchToEigen<Tensor3dXf, 3>(prediction_tensors.detach());
        SimpleEncoderDecoderLSTM simple_encoder_decoder_model, loaded_model;
        if (!simple_encoder_decoder_model.load_from_jit_module(model)) {
            throw std::runtime_error("Couldn't load model from jit.\n");
        }
        std::ios::sync_with_stdio(false);
        std::ofstream output_file(base_path / "data.txt");
        if (!output_file) {
            throw std::runtime_error("Error opening data.txt file!");
        }
        simple_encoder_decoder_model.load_to_file(output_file);
        if (!output_file.good()) {
            throw std::runtime_error("Error writing to file!");
        }
        output_file.close();

        std::ifstream input_file(base_path / "data.txt");
        if (!input_file) {
            throw std::runtime_error("Error opening data.txt file!");
        }
        loaded_model.load_from_file(input_file);
        input_file.close();

        if (!simple_encoder_decoder_model.IsEqual(loaded_model, 1e-5)) {
            throw std::runtime_error(
                "Model is not correctly loaded. The Max difference between "
                "their "
                "elements: " +
                std::to_string(simple_encoder_decoder_model.MaxAbsDifference(
                    loaded_model)));
        }
        if (!TestIfEqual<Tensor3dXf>(
                prediction, simple_encoder_decoder_model.forward(input))) {
            throw std::runtime_error(
                "Error: Comparison of our prediction and known output failed."
                "The Absolute difference is: " +
                std::to_string(MaxAbsDifference(
                    prediction, simple_encoder_decoder_model.forward(input))));
        }
    }
    catch (const std::exception &e) {
        std::cout << "SimpleEncoderDecoderLSTM Model Test not passed: \n"
                  << "Error occurred in file: " << __FILE__
                  << " at line: " << __LINE__ << "\n"
                  << e.what() << std::endl;
        return;
    }
    std::cout << "SimpleEncoderDecoderLSTM Model Test Successfully passed"
              << std::endl;
}

void TestDNS48()
{
    try {
        fs::path base_path = "../tests/test_data/dns48";
        fs::path input_path = base_path / "input.pth";
        fs::path model_path = base_path / "model.pth";
        fs::path prediction_path = base_path / "prediction.pth";
        assert(fs::exists(input_path) && "Input path does not exist");
        assert(fs::exists(model_path) && "Model path does not exist");
        assert(fs::exists(prediction_path) && "Prediction path does not exist");
        torch::jit::script::Module prior_input, model, prior_prediction;
        torch::Tensor input_tensors, prediction_tensors;

        try {
            model = torch::jit::load(model_path);
            prior_input = torch::jit::load(input_path);
            prior_prediction = torch::jit::load(prediction_path);
            input_tensors = prior_input.attr("prior").toTensor();
            prediction_tensors = prior_prediction.attr("prior").toTensor();
            std::cout << "DNS48 Model loaded successfully" << std::endl;
        }
        catch (const c10::Error &e) {
            throw std::runtime_error("Error loading the model: " +
                                     std::string(e.what()));
        }
        assert(input_tensors.dim() == 3 && "Input tensor must be 3D");
        assert(prediction_tensors.dim() == 3 && "Prediction tensor must be 3D");
        Tensor3dXf input = TorchToEigen<Tensor3dXf, 3>(input_tensors);
        Tensor3dXf prediction = TorchToEigen<Tensor3dXf, 3>(prediction_tensors.detach()); 
        DemucsModel demucs_model, loaded_model;
        if (!demucs_model.load_from_jit_module(model)) {
            throw std::runtime_error("Couldn't load model from jit.\n");
        }
        std::ios::sync_with_stdio(false);
        std::ofstream output_file(base_path / "data.txt");
        if (!output_file) {
            throw std::runtime_error("Error opening data.txt file!");
        }
        demucs_model.load_to_file(output_file);
        if (!output_file.good()) {
            throw std::runtime_error("Error writing to file!");
        }
        output_file.close();
        std::ifstream input_file(base_path / "data.txt");
        if (!input_file) {
            throw std::runtime_error("Error opening data.txt file!");
        }
        loaded_model.load_from_file(input_file);
        input_file.close();
        if (!demucs_model.IsEqual(loaded_model, 1e-5)) {
            throw std::runtime_error(
                "Model is not correctly loaded. The Max difference between "
                "their "
                "elements: " +
                std::to_string(demucs_model.MaxAbsDifference(loaded_model)));
        }
        if (!TestIfEqual<Tensor3dXf>(prediction, demucs_model.forward(input))) {
            throw std::runtime_error(
                "Error: Comparison of our prediction and known output failed."
                "The Absolute difference is: " +
                std::to_string(
                    MaxAbsDifference(prediction, demucs_model.forward(input))));
        }
    }
    catch (const std::exception &e) {
        std::cout << "DNS48 Model Test not passed: \n"
                  << "Error occurred in file: " << __FILE__
                  << " at line: " << __LINE__ << "\n"
                  << e.what() << std::endl;
        return;
    }
    std::cout << "DNS48 Model Test Successfully passed" << std::endl;
}

void TestDemucsFullForwardAudio()
{
    fs::path input_dir = "../dataset/noisy";
    fs::path output_dir = "../dataset/test_demucs_full";
    DemucsModel demucs_model;
    fs::path model_path = "../tests/test_data/dns48";
    std::ifstream input_file(model_path / "data.txt");
    if (!input_file) {
        std::cerr << "Unable to open model file." << std::endl;
        return;
    }
    try {
        demucs_model.load_from_file(input_file);
        for (const auto &entry : fs::directory_iterator(input_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".wav") {
                const fs::path input_file_path = entry.path();
                const fs::path output_file_path =
                    output_dir / input_file_path.filename();
                try {
                    Tensor2dXf wav = ReadAudioFromFile(input_file_path);
                    Tensor3dXf big_wav(1, wav.dimension(0), wav.dimension(1));
                    big_wav.chip(0, 0) = wav;
                    Tensor3dXf ans = demucs_model.forward(big_wav);
                    WriteAudioFromFile(output_file_path, ans.chip(0, 0));
                    std::cout << "Processed: " << input_file_path << " -> "
                              << output_file_path << std::endl;
                }
                catch (const std::exception &e) {
                    throw std::runtime_error("Model Processing " +
                                             std::string(input_file_path) +
                                             ": " + e.what() + "\n");
                }
            }
        }
    }
    catch (const std::exception &e) {
        std::cout << "DemucsFullForwardAudio Test not passed: \n"
                  << "Error occurred in file: " << __FILE__
                  << " at line: " << __LINE__ << "\n"
                  << e.what() << std::endl;
        return;
    }
    std::cout << "TestDemucsFullForwardAudio Test Successfully passed"
              << std::endl;
}

void TestStreamerForwardAudio()
{
    fs::path input_dir = "../dataset/noisy";
    fs::path output_dir = "../dataset/test_streamer";
    auto start = std::chrono::high_resolution_clock::now();
    DemucsStreamer streamer;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Streamer creation time is: " << duration.count() << "s"
              << std::endl;
    int sample_rate;
    try {
        for (const auto &entry : fs::directory_iterator(input_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".wav") {
                const fs::path input_file_path = entry.path();
                const fs::path output_file_path =
                    output_dir / input_file_path.filename();
                try {
                    Tensor2dXf wav =
                        ReadAudioFromFile(input_file_path, &sample_rate);
                    start = std::chrono::high_resolution_clock::now();
                    Tensor2dXf ans = streamer.forward(wav);
                    end = std::chrono::high_resolution_clock::now();
                    duration =
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            end - start);
                    WriteAudioFromFile(output_file_path, ans, &sample_rate);
                    std::cout << "Processed: " << input_file_path << " -> "
                              << output_file_path << std::endl;
                }
                catch (const std::exception &e) {
                    throw std::runtime_error("Model Processing " +
                                             std::string(input_file_path) +
                                             ": " + e.what() + "\n");
                }
            }
        }
    }
    catch (const std::exception &e) {
        std::cout << "StreamerForwardAudio Test not passed: \n"
                  << "Error occurred in file: " << __FILE__
                  << " at line: " << __LINE__ << "\n"
                  << e.what() << std::endl;
        return;
    }
    std::cout << "StreamerForwardAudio Test Successfully passed" << std::endl;
}

void GenerateTestsforAudio()
{
    fs::path input_dir = "../dataset/noisy";
    fs::path output_dir = "../dataset/test_sdr";
    fs::path clean_dir = "../dataset/clean";
    fs::path debug_dir = "../dataset/debug";
    try {
        for (int stride = 4096 * 4; stride >= 128; stride /= 2) {
            DemucsStreamer streamer(stride);
            float mean_sdr = 0;
            float mean_rtf = 0;
            int sdr_counter = 0;
            int mean_counter = 0;
            for (const auto &entry : fs::directory_iterator(input_dir)) {
                if (entry.is_regular_file() &&
                    entry.path().extension() == ".wav") {
                    const fs::path input_file_path = entry.path();
                    const fs::path output_file_path =
                        output_dir / input_file_path.filename();
                    const fs::path clean_file_path =
                        clean_dir / input_file_path.filename();
                    const fs::path debug_file_path =
                        debug_dir / input_file_path.filename();
                    try {
                        double length_in_seconds;
                        int sample_rate;
                        Tensor2dXf wav = ReadAudioFromFile(
                            input_file_path, &sample_rate, &length_in_seconds);
                        Tensor2dXf clean_wav =
                            ReadAudioFromFile(clean_file_path);
                        auto start = std::chrono::high_resolution_clock::now();
                        Tensor2dXf ans = streamer.forward(wav);
                        auto end = std::chrono::high_resolution_clock::now();
                        auto duration =
                            std::chrono::duration_cast<
                                std::chrono::milliseconds>(end - start)
                                .count();
                        Tensor2dXf resampled_ans = WriteAudioFromFile(
                            output_file_path, ans, &sample_rate);
                        Tensor2dXf resampled_clean_wav = WriteAudioFromFile(
                            debug_file_path, clean_wav, &sample_rate);
                        mean_rtf += static_cast<double>(duration) /
                                    (1000.0 * length_in_seconds);
                        float sdr =
                            compare_audio(resampled_ans, resampled_clean_wav);
                        mean_sdr += sdr;
                        sdr_counter++;
                        mean_counter++;
                        std::cout
                            << "Current Mean SDR is:" << mean_sdr / mean_counter
                            << " Mean RTF is: " << mean_rtf / sdr_counter
                            << std::endl;
                    }
                    catch (const std::exception &e) {
                        std::cerr << "Model Processing " << input_file_path
                                  << ": " << e.what() << std::endl;
                    }
                }
            }

            mean_sdr /= mean_counter;
            mean_rtf /= sdr_counter;
            std::cout << " numThreads:" << 10 << " stride:" << stride
                      << "Mean SDR is:" << mean_sdr
                      << " Mean RTF is: " << mean_rtf << std::endl;
        }
    }
    catch (const std::exception &e) {
        std::cout << "GenerateTestsforAudio not passed: \n"
                  << "Error occurred in file: " << __FILE__
                  << " at line: " << __LINE__ << "\n"
                  << e.what() << std::endl;
        return;
    }
    std::cout << "GenerateTestsforAudio Successfully passed" << std::endl;
}

void TestResampler()
{
    try {
        fs::path input_dir = "../dataset/noisy";
        fs::path output_dir = "../dataset/test_resample";
        int sample_rate;
        Tensor2dXf wav = ReadAudioFromFile(input_dir / "001.wav", &sample_rate);
        WriteAudioFromFile(output_dir / "001.wav", wav, &sample_rate);
    }
    catch (const std::exception &e) {
        std::cout << "TestResampler not passed: \n"
                  << "Error occurred in file: " << __FILE__
                  << " at line: " << __LINE__ << "\n"
                  << e.what() << std::endl;
        return;
    }
    std::cout << "TestResampler Successfully passed" << std::endl;
}