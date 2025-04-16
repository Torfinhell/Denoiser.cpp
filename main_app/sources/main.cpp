#include "audio.h"
#include "layers.h"
#include "tests.h"
#include <exception>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;
int main(int argc, char *argv[])
{

    if (argc < 3 || argc > 5) {
        std::cerr << "Error: Build with arguments: " << argv[0]
                  << " input_dir output_dir stride(int, optional) threads(int, "
                     "optional)"
                  << std::endl;
        return 1;
    }

    const fs::path input_dir = argv[1];
    const fs::path output_dir = argv[2];

    if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
        std::cerr << "Error: The path '" << input_dir
                  << "' does not exist or is not a folder." << std::endl;
        return 1;
    }

    if (!fs::exists(output_dir)) {
        if (!fs::create_directory(output_dir)) {
            std::cerr << "Error: Unable to create output directory '"
                      << output_dir << "'." << std::endl;
            return 1;
        }
    }
    int stride = 4096, numThreads = 10;
    if (argc == 4) {
        stride = std::atoi(argv[3]);
    }
    if (argc == 5) {
        numThreads = std::atoi(argv[4]);
    }

    if (stride <= 0) {
        std::cerr << "Error: Stride must be a positive integer." << std::endl;
        return 1;
    }

    if (numThreads <= 0) {
        std::cerr << "Error: Threads must be a positive integer." << std::endl;
        return 1;
    }
    fs::path model_path = "../tests/test_data/dns48";
    if (!fs::exists(model_path / "data.txt")) {
        fs::path base_path = "../tests/test_data/dns48";
        fs::path model_path = base_path / "model.pth";
        if (!fs::exists(model_path)) {
            std::cerr << "Model path does not exist. Tru running "
                         "tests/create_tests.py to create the model file"<<std::endl;
            return 1;
        }
        torch::jit::script::Module model;
        try {
            model = torch::jit::load(model_path);
            std::cout << "DNS48 Model loaded successfully" << std::endl;
        }
        catch (const std::exception &e) {
            std::cerr << "Error: loading the model from model.pth: " +
                             std::string(e.what());
            return 1;
        }
        try {
            DemucsModel demucs_model, loaded_model;
            if (!demucs_model.load_from_jit_module(model)) {
                throw std::runtime_error("Couldn't load model from jit.\n");
            }
            std::ios::sync_with_stdio(false);
            std::ofstream output_file(base_path / "data.txt");
            if (!output_file) {
                throw std::runtime_error("Error: opening data.txt file!");
            }
            demucs_model.load_to_file(output_file);
            if (!output_file.good()) {
                throw std::runtime_error("Error: writing to file!");
            }
            output_file.close();
        }
        catch (const std::exception &e) {
            std::cerr << "Error: loading the model to data.txt: " +
                             std::string(e.what())<<std::endl;
            return 1;
        }
    }
    DemucsStreamer streamer;
    try {
        for (const auto &entry : fs::directory_iterator(input_dir)) {
            if (entry.is_regular_file() &&
                (entry.path().extension() == ".wav" ||
                 entry.path().extension() == ".mp3")) {
                const fs::path input_file_path = entry.path();
                const fs::path output_file_path =
                    output_dir / input_file_path.filename();
                try {
                    double length_in_seconds;
                        int sample_rate;
                    Tensor2dXf wav = ReadAudioFromFile(input_file_path,&sample_rate, &length_in_seconds);
                    auto start = std::chrono::high_resolution_clock::now();
                    Tensor2dXf ans = streamer.forward(wav);
                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration =
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            end - start)
                            .count();

                    WriteAudioFromFile(output_file_path, ans, &sample_rate);
                    std::cout << "Duration: " << duration
                              << " milliseconds, Length of Audio is:"<<length_in_seconds<<"s, Processed: " << input_file_path
                              << " -> " << output_file_path << std::endl;
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
        std::cout << "Denoiser couldnt denoise the folder " << input_dir
                  << ": \n"
                  << "Error: occurred in file: " << __FILE__ << "\n"
                  << e.what() << std::endl;
        return 1;
    }
    std::cout << "Denoiser successfully denoised  files in folder: "<<input_dir << std::endl;

    return 0;
}