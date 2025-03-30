#include "audio.h"
#include "tensors.h"
#include <array>
#include <iostream>
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
    for (int i = 0; i + frame_size < audio.dimension(2); i += frame_size) {
        std::array<long, 3> offset = {0, 0, i};
        std::array<long, 3> extent = {audio.dimension(0), audio.dimension(1),
                                      frame_size};
        ret.emplace_back(audio.slice(offset, extent));
        frame_size = total_stride;
    }
    return ret;
}
