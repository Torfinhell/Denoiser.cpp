#include "tests.h"
#include "layers.h"
#include "tensors.h"
#include <chrono>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <ios>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
std::array<long, 1> indices_dim_1 = {0};
std::array<long, 2> indices_dim_2 = {0, 0};
std::array<long, 3> indices_dim_3 = {0, 0, 0};
std::array<long, 4> indices_dim_4 = {0, 0, 0, 0};
namespace fs = std::filesystem;
using namespace Eigen;
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
        Tensor1dXf prediction = TorchToEigen<Tensor1dXf, 1>(prediction_tensors);
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
        Tensor3dXf prediction = TorchToEigen<Tensor3dXf, 3>(prediction_tensors);
        OneEncoder one_encoder_model, loaded_model;
        // print_model_layers(model);
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
        Tensor3dXf prediction = TorchToEigen<Tensor3dXf, 3>(prediction_tensors);
        SimpleEncoderDecoder simple_encoder_decoder_model, loaded_model;
        // print_model_layers(model);
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
        Tensor3dXf prediction = TorchToEigen<Tensor3dXf, 3>(prediction_tensors);
        SimpleEncoderDecoderLSTM simple_encoder_decoder_model, loaded_model;
        // print_model_layers(model);
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

void TestBasicDemucsModel()
{
    try {
        fs::path base_path = "../tests/test_data/BasicDemucs";
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
            std::cout << "DemucsModel Model loaded successfully" << std::endl;
        }
        catch (const c10::Error &e) {
            throw std::runtime_error("Error loading the model: " +
                                     std::string(e.what()));
        }
        assert(input_tensors.dim() == 3 && "Input tensor must be 3D");
        assert(prediction_tensors.dim() == 3 && "Prediction tensor must be 3D");
        // print_model_layers(model);
        Tensor3dXf input = TorchToEigen<Tensor3dXf, 3>(input_tensors);
        Tensor3dXf prediction = TorchToEigen<Tensor3dXf, 3>(prediction_tensors);
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
            std::ofstream file1("data1.txt");
            file1<<prediction;
            throw std::runtime_error(
                "Error: Comparison of our prediction and known output failed."
                "The Absolute difference is: " +
                std::to_string(
                    MaxAbsDifference(prediction, demucs_model.forward(input))));
        }
    }
    catch (const std::exception &e) {
        std::cout << "DemucsModel Model Test not passed: \n"
                  << "Error occurred in file: " << __FILE__
                  << " at line: " << __LINE__ << "\n"
                  << e.what() << std::endl;
        return;
    }
    std::cout << "DemucsModel Model Test Successfully passed" << std::endl;
}