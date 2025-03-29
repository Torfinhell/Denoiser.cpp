#include "audio.h"
#include "tensors.h"
#include <array>
#include <vector>

Tensor3dXf ReadAudioFromFile(fs::path file_name)
{
    Tensor3dXf tensor;
    return tensor;
}
void WriteAudioFromFile(fs::path file_name, Tensor3dXf tensor)
{
    return;
}
std::vector<Tensor3dXf> SplitAudio(Tensor3dXf audio, int total_length,
                                   int total_stride)
{
    std::vector<Tensor3dXf> ret;
    int frame_size = total_length;
    for (int i = 0; i < audio.dimension(2); i += frame_size) {
        std::array<long, 3> offset = {0, 0, 0};
        std::array<long, 3> extent = {audio.dimension(0), audio.dimension(1),
                                      (audio.dimension(2) < frame_size)
                                          ? audio.dimension(2)
                                          : frame_size};
        ret.emplace_back(audio.slice(offset, extent));
        offset[2] += frame_size;
        extent[2] = frame_size;
        frame_size = total_stride;
    }
    return ret;
}
std::vector<Tensor3dXf> SplitAudio(Tensor2dXf audio, int total_length,
                                   int total_stride)
{
    Tensor3dXf tensor(1, audio.dimension(0), audio.dimension(1));
    tensor.chip(0, 0) = audio;
    return SplitAudio(tensor, total_length, total_stride);
}
Tensor2dXf CombineAudio(std::vector<Tensor3dXf> audio)
{
    Tensor2dXf tensor;
    return tensor;
}