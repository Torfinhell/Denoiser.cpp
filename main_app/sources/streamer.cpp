#include "audio.h"
#include "layers.h"
#include <iostream>

Tensor3dXf Concat3DLast(std::vector<Tensor3dXf> vec){
    Eigen::MatrixXf ret_audio;
    for (int i = 0; i < vec.size(); i++) {
        int height = vec[i].dimension(1);
        int width = vec[i].dimension(2);

        Eigen::MatrixXf matrix = Eigen::Map<Eigen::MatrixXf>(
            vec[i].data(), height, width);
        if (i == 0) {
            ret_audio = matrix;
        }
        else {
            ret_audio.conservativeResize(Eigen::NoChange,
                                         ret_audio.cols() + matrix.cols());
            ret_audio.rightCols(matrix.cols()) = matrix;
        }
    }
    Tensor3dXf final_tensor = Eigen::TensorMap<Tensor3dXf>(
        ret_audio.data(), 1,ret_audio.rows(), ret_audio.cols());
    return final_tensor;
}
Tensor2dXf DemucsStreamer::forward(Tensor2dXf wav)
{
    Tensor3dXf bigwav(1, wav.dimension(0), wav.dimension(1));
    bigwav.chip(0, 0) = wav;
    std::vector<Tensor3dXf> buffer_audio;
    int frame_size = total_length;
    int i = 0;
    while (i + frame_size < bigwav.dimension(2)) {
        std::array<long, 3> offset = {0, 0, i};
        std::array<long, 3> extent = {bigwav.dimension(0), bigwav.dimension(1),
                                      frame_size};
        buffer_audio.emplace_back(bigwav.slice(offset, extent));
        i += frame_size;
        frame_size = total_stride;
    }
    std::vector<Tensor3dXf> return_audio(buffer_audio.size());
    const int batch_size = 2;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < buffer_audio.size(); i++) {
        return_audio[i] = feed(buffer_audio[i]);
    }
    Tensor3dXf outs_rt= Concat3DLast(return_audio);
    return outs_rt.chip(0, 0);
}

Tensor3dXf DemucsStreamer::feed(Tensor3dXf wav)
{
    assert(wav.dimension(0) == demucs_model.chin);
    pending = Concat3DLast({pending,wav});
    std::vector<Tensor3dXf> outs;
    int last_i = 0;
    std::array<long, 3> offset;
    std::array<long, 3> extent;
    assert(stride > 0 && "stride is positive");
    for (int i = 0; i + total_length <= pending.dimension(2);
         i += stride, last_i += stride) {
        frames++;
        offset = {0, 0, i};
        extent = {pending.dimension(0), pending.dimension(1), total_length};
        outs.emplace_back(demucs_model.forward(
            pending.slice(offset, extent), this)); // parallelize
    }
    offset = {0, 0, last_i};
    extent = {pending.dimension(0), pending.dimension(1),
              pending.dimension(2) - last_i};
    pending = Tensor3dXf{pending.slice(offset, extent)};
    Tensor3dXf outs_answer=Concat3DLast(outs);
    return outs_answer;
}



