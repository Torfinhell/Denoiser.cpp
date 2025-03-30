#include "audio.h"
#include "layers.h"
#include "tensors.h"
#include "tests.h"
#include <mutex>
#include <vector>
int main()
{
    // TestSimpleModel();
    // TestOneEncoder();
    // TestSimpleEncoderDecoder();
    // TestSimpleEncoderDecoderLSTM();
    // TestBasicDemucsModel();
    TestDemucsStreamer();
    // поменять форвард
    //  std::string wav_name="p287_001.wav";
    //  fs::path path_to_wav=fs::path("../data/debug/noisy") / wav_name;
    //  fs::path ret_path_wav=fs::path("../data/debug/answer") / wav_name;
    //  Tensor3dXf read_audio=ReadAudioFromFile(path_to_wav);
    //  std::vector<Tensor3dXf> buffer_audio=SplitAudio(read_audio);
    //  std::vector<Tensor3dXf> return_audio(buffer_audio.size());
    //  const int numThreads=4;
    //  const int resample_buffer=256,resample_lookahead=64,dry=0,num_frames=1;
    //  std::vector<std::mutex> mutexes;
    //  std::vector<std::mutex> lstm_mutexes;
    //  std::vector<std::unique_lock<std::mutex>> lstm_locks(numThreads);
    //  std::vector<std::thread> threads(numThreads);
    //  const int batch_size=2;
    //  DemucsModel demucs_model;
    //  LstmState lstm_state;
    //  for(int i=0;i<numThreads;i++){
    //      lstm_locks.emplace_back(std::unique_lock<std::mutex>
    //      {lstm_mutexes[i]});
    //  }
    //  lstm_locks[0].unlock();
    //  int buffer_index=0;
    //  while (buffer_index < buffer_audio.size()) {
    //      std::thread t([&return_audio,&buffer_audio, &lstm_state, numThreads,
    //      buffer_index, &lstm_mutexes, &lstm_locks, &demucs_model]() {
    //          int thread_ind = buffer_index % numThreads;
    //          auto start = std::chrono::high_resolution_clock::now();
    //          {
    //              std::lock_guard<std::mutex> lock(lstm_mutexes[thread_ind]);
    //              return_audio[buffer_index]=demucs_model.forward(buffer_audio[buffer_index],
    //              lstm_locks, thread_ind, lstm_state,numThreads);
    //          }
    //          auto end = std::chrono::high_resolution_clock::now();
    //          std::chrono::duration<double, std::milli> duration = end -
    //          start; std::cout << "Thread ID: " << thread_ind << ", Buffer
    //          Index: " << buffer_index << ", Time taken: " << duration.count()
    //          << " ms" << std::endl;
    //      });
    //      t.join();
    //      buffer_index++;
    //  }
    //  Tensor3dXf ret_audio=CombineAudio(return_audio);
    //  WriteAudioFromFile(ret_path_wav,ret_audio);
}