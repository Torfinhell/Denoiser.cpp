#include "tensors.h"
#include <caffe2/serialize/read_adapter_interface.h>
#include <filesystem>
#include <iostream>
namespace fs = std::filesystem;
using namespace Tensors;
Tensor2dXf ReadAudioFromFile(fs::path file_name);
void WriteAudioFromFile(fs::path file_name, Tensor2dXf tensor);
