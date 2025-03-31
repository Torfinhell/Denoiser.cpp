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
    int i=0;
    while(i + frame_size < audio.dimension(2)){
        std::array<long, 3> offset = {0, 0, i};
        std::array<long, 3> extent = {audio.dimension(0), audio.dimension(1),
                                      frame_size};
        ret.emplace_back(audio.slice(offset, extent));
        i+=frame_size;
        frame_size = total_stride;
    }
    return ret;
}

// std::vector<Tensor3dXf> SplitAudio(Tensor3dXf audio, int total_length,
//     int total_stride)
// {
// std::vector<Tensor3dXf> ret;
// int offset = 0;
// const int audio_length = audio.dimension(2);

// if (total_length <= audio_length) {
// std::array<long, 3> offset_arr = {0, 0, 0};
// std::array<long, 3> extent = {audio.dimension(0), audio.dimension(1),
//        total_length};
// ret.emplace_back(audio.slice(offset_arr, extent));
// offset += total_length;
// }

// while (offset + total_stride <= audio_length) {
// std::array<long, 3> offset_arr = {0, 0, offset};
// std::array<long, 3> extent = {audio.dimension(0), audio.dimension(1),
//        total_stride};
// ret.emplace_back(audio.slice(offset_arr, extent));
// offset += total_stride;
// }

// int remaining = audio_length - offset;
// if (remaining > 0) {
// std::array<long, 3> offset_arr = {0, 0, offset};
// std::array<long, 3> extent = {audio.dimension(0), audio.dimension(1),
//        remaining};
// ret.emplace_back(audio.slice(offset_arr, extent));
// }

// return ret;
// }
