#include "tensors.h"
#include <caffe2/serialize/read_adapter_interface.h>
#include <filesystem>
namespace fs = std::filesystem;
using namespace Tensors;
Tensor3dXf ReadAudioFromFile(fs::path file_name);
std::vector<Tensor3dXf> SplitAudio(Tensor3dXf audio, int total_length,
                                   int total_stride);
std::vector<Tensor3dXf> SplitAudio(Tensor2dXf audio, int total_length,
                                   int total_stride);
Tensor2dXf CombineAudio(std::vector<Tensor3dXf> audio);
void WriteAudioFromFile(fs::path file_name, Tensor3dXf tensor);