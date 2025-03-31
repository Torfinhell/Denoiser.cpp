#include "layers.h"
#include "audio.h"

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
Tensor2dXf DemucsStreamer::forward(Tensor2dXf wav)
{
    Tensor3dXf bigwav(1, wav.dimension(0), wav.dimension(1));
    bigwav.chip(0, 0) = wav;
    std::vector<Tensor3dXf> buffer_audio =
        SplitAudio(bigwav, total_length, total_stride);
    std::vector<Tensor3dXf> return_audio(buffer_audio.size());
    const int batch_size = 2;
    LstmState lstm_state;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < buffer_audio.size(); i++) {
        return_audio[i] = feed(buffer_audio[i], lstm_state);
    }
    Tensor3dXf outs_rt(1, demucs_model.chin, 0);
    for (const Tensor3dXf &tensor : return_audio) {
        outs_rt = Tensor3dXf{outs_rt.concatenate(tensor, 2)};
    }
    return outs_rt.chip(0, 0);
}

Tensor3dXf DemucsStreamer::feed(Tensor3dXf wav, LstmState &lstm_state)
{
    assert(wav.dimension(0) == demucs_model.chin);
    pending = Tensor3dXf{pending.concatenate(wav, 2).eval()};
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
            pending.slice(offset, extent), lstm_state, this)); // parallelize
    }
    offset = {0, 0, last_i};
    extent = {pending.dimension(0), pending.dimension(1),
              pending.dimension(2) - last_i};
    pending = Tensor3dXf{pending.slice(offset, extent)};
    Tensor3dXf outs_answer(1, demucs_model.chin, 0);
    if (!outs.empty()) {
        for (const Tensor3dXf &tensor : outs) {
            outs_answer = Tensor3dXf{outs_answer.concatenate(tensor, 2)};
        }
    }
    return outs_answer;
}

Tensor2dXf DemucsStreamer::forward_regular(Tensor2dXf wav)
{
    DemucsModel demucs_model;
    fs::path base_path = "../tests/test_data/dns48";
    std::ifstream input_file(base_path / "data.txt");
    demucs_model.load_from_file(input_file);
    const int resample_lookahead = 64;
    const int stride = 4;
    const int depth = 5;
    const int resample=4;
    const int total_stride = (pow(stride,depth))/resample;
    const int num_frames = 1;
    const int frame_length =
        demucs_model.valid_length(1);
    const int total_length = frame_length + resample_lookahead;
    const int numThreads = 6;
    const int resample_buffer = 256;
    Tensor3dXf big_wav(1, wav.dimension(0), wav.dimension(1));
    big_wav.chip(0, 0) = wav;
    std::vector<Tensor3dXf> buffer_audio =
        SplitAudio(big_wav, total_stride, total_stride);/////////////
    std::vector<Tensor3dXf> return_audio(buffer_audio.size());
    const int batch_size = 2;
    LstmState lstm_state;
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0;i<buffer_audio.size();i++){
        return_audio[i]=demucs_model.forward(buffer_audio[i],lstm_state);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << ", Time taken: " << duration.count() << "s" << std::endl;
    Tensor3dXf ret_audio(1, demucs_model.chin, 0);
    for(int i=0;i<return_audio.size();i++){
        ret_audio=Tensor3dXf{ret_audio.concatenate(return_audio[i], 2)};
    }
    return ret_audio.chip(0,0);
}