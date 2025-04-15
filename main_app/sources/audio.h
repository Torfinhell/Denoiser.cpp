#include "tensors.h"
#include <caffe2/serialize/read_adapter_interface.h>
#include <filesystem>
#include <iostream>
namespace fs = std::filesystem;
using namespace Tensors;
Tensor2dXf ReadAudioFromFile(fs::path file_name, int* sample_rate = nullptr, double *length_in_seconds = nullptr);
void WriteAudioFromFile(fs::path file_name, Tensor2dXf wav, int* sample_rate = nullptr);

float compare_audio(Tensor2dXf noisy, Tensor2dXf clean);

Tensor2dXf ResampleAudio(Tensor2dXf big_wav, int input_sample_rate, int output_sample_rate);
