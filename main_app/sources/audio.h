#include "tensors.h"
#include <caffe2/serialize/read_adapter_interface.h>
#include <filesystem>
namespace fs = std::filesystem;
using namespace Tensors;
Tensor3dXf ReadAudioFromFile(fs::path file_name);
std::vector<Tensor3dXf> SplitAudio(Tensor3dXf audio, int total_length,
                                   int total_stride);
void WriteAudioFromFile(fs::path file_name, Tensor3dXf tensor);
template <typename Tensor> Tensor ConcatList(std::vector<Tensor> vec, int dim)
{
    int sumdim = 0;
    for (const Tensor &tensor : vec) {
        sumdim += tensor.dimension(dim);
    }
    auto shape = vec[0].dimensions();
    shape[dim] = sumdim;
    Tensor concat_tensor(shape);
    int counter = 0;
    for (const Tensor &tensor : vec) {
        for (int i = 0; i < tensor.dimension(dim); i++) {
            concat_tensor.chip(counter++, dim) = tensor.chip(i, dim);
        }
    }
    return concat_tensor;
}
