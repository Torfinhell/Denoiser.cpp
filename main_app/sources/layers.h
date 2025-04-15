#pragma once
#include "tensors.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <torch/script.h>

#include <vector>
#define EIGEN_USE_MKL_ALL
#define EIGEN_USE_BLAS
#define EIGEN_VECTORIZE_SSE4_2
#define PI 3.141592653589793

extern Eigen::array<Eigen::IndexPair<int>, 1> product_dims_reg;
extern Eigen::array<Eigen::IndexPair<int>, 1> product_dims_sec_transposed;
extern Eigen::array<Eigen::IndexPair<int>, 1> product_dims_first_transposed;
extern Eigen::array<Eigen::IndexPair<int>, 1> product_dims_both_transposed;
extern std::array<long, 1> indices_dim_1;
extern std::array<long, 2> indices_dim_2;
extern std::array<long, 3> indices_dim_3;
extern std::array<long, 4> indices_dim_4;
using namespace Tensors;
template <typename T> T max_of_multiple(std::initializer_list<T> values)
{
    return *std::max_element(values.begin(), values.end());
}
template <typename EigenTensor, int NumDim>
void WriteTensor(EigenTensor &tensor, std::ofstream &outputstream,
                 std::array<long, NumDim> &indices, int current_pos = 0)
{
    if (current_pos == 0) {
        for (int i = 0; i < NumDim; i++) {
            outputstream << tensor.dimension(i) << " ";
        }
    }
    if (current_pos == NumDim) {
        outputstream << tensor(indices) << " ";
    }
    else {
        for (long long i = 0; i < tensor.dimension(current_pos); i++) {
            indices[current_pos] = i;
            WriteTensor<EigenTensor, NumDim>(tensor, outputstream, indices,
                                             current_pos + 1);
        }
    }
}

template <typename EigenTensor, int NumDim>
void ReadTensor(EigenTensor &tensor, std::ifstream &inputstream,
                std::array<long, NumDim> &indices, int current_pos = 0)
{
    std::array<long, NumDim> dimensions;
    if (current_pos == 0) {
        for (int i = 0; i < NumDim; i++) {
            inputstream >> dimensions[i];
        }
        tensor.resize(dimensions);
    }
    if (current_pos == NumDim) {
        inputstream >> tensor(indices);
    }
    else {
        for (long long i = 0; i < tensor.dimension(current_pos); i++) {
            indices[current_pos] = i;
            ReadTensor<EigenTensor, NumDim>(tensor, inputstream, indices,
                                            current_pos + 1);
        }
    }
}
struct SimpleModel {
    void forward(Tensor1dXf tensor);
    bool load_from_jit_module(const torch::jit::script::Module &module);
    void load_to_file(std::ofstream &outputstream);
    void load_from_file(std::ifstream &inputstream);
    float MaxAbsDifference(const SimpleModel &other) const;
    bool IsEqual(const SimpleModel &other, float tolerance = 1e-5)const;
    ~SimpleModel() {}

    Tensor2dXf fc_weights;
    Tensor1dXf fc_bias;
    Tensor1dXf result;
};
struct Conv1D {
    Tensor3dXf forward(Tensor3dXf tensor, int InputChannels, int OutputChannels,
                       int kernel_size = 3, int stride = 1, int padding = 0);
    bool load_from_jit_module(const torch::jit::script::Module &module,
                              std::string weights_index);
    void load_to_file(std::ofstream &outputstream);
    void load_from_file(std::ifstream &inputstream);
    float MaxAbsDifference(const Conv1D &other) const;
    bool IsEqual(const Conv1D &other, float tolerance = 1e-5)const;
    ~Conv1D() {}
    int GetSize(int size, int kernel_size, int stride);
    Tensor3dXf conv_weights;
    Tensor1dXf conv_bias;
};
struct ConvTranspose1d {
    Tensor3dXf forward(Tensor3dXf tensor, int InputChannels, int OutputChannels,
                       int kernel_size = 3, int stride = 1);
    bool load_from_jit_module(const torch::jit::script::Module &module,
                              std::string weights_index);
    void load_to_file(std::ofstream &outputstream);
    void load_from_file(std::ifstream &inputstream);
    float MaxAbsDifference(const ConvTranspose1d &other) const;
    bool IsEqual(const ConvTranspose1d &other, float tolerance = 1e-5)const;
    ~ConvTranspose1d() {}
    int GetTransposedSize(int size, int kernel_size, int stride);
    Tensor3dXf conv_tr_weights;
    Tensor1dXf conv_tr_bias;
};
struct RELU {
    Tensor3dXf forward(Tensor3dXf tensor);
};
struct GLU {
    Tensor3dXf forward(Tensor3dXf tensor);
};
struct OneEncoder {
    Tensor3dXf forward(Tensor3dXf tensor);
    bool load_from_jit_module(const torch::jit::script::Module &module,
                              std::string weights_index = "0");
    void load_to_file(std::ofstream &outputstream);
    void load_from_file(std::ifstream &inputstream);
    float MaxAbsDifference(const OneEncoder &other) const;
    bool IsEqual(const OneEncoder &other, float tolerance = 1e-5) const;
    ~OneEncoder() {}
    Conv1D conv_1_1d;
    Conv1D conv_2_1d;
    RELU relu;
    GLU glu;
    int hidden, ch_scale, kernel_size, stride, chout, chin;
    OneEncoder(int hidden = 48, int ch_scale = 2, int kernel_size = 8,
               int stride = 4, int chout = 1, int chin = 1);
};

struct OneDecoder {
    Tensor3dXf forward(Tensor3dXf tensor);
    bool load_from_jit_module(const torch::jit::script::Module &module,
                              std::string weights_index = "0");
    void load_to_file(std::ofstream &outputstream);
    void load_from_file(std::ifstream &inputstream);
    float MaxAbsDifference(const OneDecoder &other) const;
    bool IsEqual(const OneDecoder &other, float tolerance = 1e-5)const;
    ~OneDecoder() {}
    Conv1D conv_1_1d;
    GLU glu;
    ConvTranspose1d conv_tr_1_1d;
    int hidden, ch_scale, kernel_size, stride, chout;
    OneDecoder(int hidden = 48, int ch_scale = 2, int kernel_size = 8,
               int stride = 4, int chout = 1);
};
struct LstmState {
    Tensor2dXf hidden_state;
    Tensor2dXf cell_state;
    bool is_created = false;
};
struct OneLSTM {
    Tensor3dXf forward(Tensor3dXf tensor, int HiddenSize, LstmState &lstm_state);
    bool load_from_jit_module(const torch::jit::script::Module &module,
                              std::string weights_index);
    void load_to_file(std::ofstream &outputstream);
    void load_from_file(std::ifstream &inputstream);
    float MaxAbsDifference(const OneLSTM &other) const;
    bool IsEqual(const OneLSTM &other, float tolerance = 1e-5)const;
    ~OneLSTM() {}
    Tensor2dXf lstm_weight_ih, lstm_weight_hh;
    Tensor2dXf lstm_bias_ih, lstm_bias_hh;
};

struct SimpleEncoderDecoder {
    Tensor3dXf forward(Tensor3dXf tensor);
    bool load_from_jit_module(const torch::jit::script::Module &module);
    void load_to_file(std::ofstream &outputstream);
    void load_from_file(std::ifstream &inputstream);
    float MaxAbsDifference(const SimpleEncoderDecoder &other) const;
    bool IsEqual(const SimpleEncoderDecoder &other, float tolerance = 1e-5)const;
    ~SimpleEncoderDecoder() {}
    OneEncoder one_encoder;
    OneDecoder one_decoder;
    int hidden, ch_scale, kernel_size, stride, chout;
    SimpleEncoderDecoder(int hidden = 48, int ch_scale = 2, int kernel_size = 8,
                         int stride = 4, int chout = 1);
};
struct SimpleEncoderDecoderLSTM {
    Tensor3dXf forward(Tensor3dXf tensor);
    bool load_from_jit_module(const torch::jit::script::Module &module);
    void load_to_file(std::ofstream &outputstream);
    void load_from_file(std::ifstream &inputstream);
    float MaxAbsDifference(const SimpleEncoderDecoderLSTM &other) const;
    bool IsEqual(const SimpleEncoderDecoderLSTM &other, float tolerance = 1e-5)const;
    ~SimpleEncoderDecoderLSTM() {}
    OneEncoder one_encoder;
    OneDecoder one_decoder;
    OneLSTM lstm1, lstm2;
    int hidden, ch_scale, kernel_size, stride, chout;
    SimpleEncoderDecoderLSTM(int hidden = 48, int ch_scale = 2,
                             int kernel_size = 8, int stride = 4,
                             int chout = 1);
};
struct DemucsStreamer;
struct DemucsModel {
    Tensor3dXf forward(Tensor3dXf mix);
    Tensor3dXf EncoderWorker(Tensor3dXf mix);
    Tensor3dXf LSTMWorker(Tensor3dXf x);
    Tensor3dXf DecoderWorker(Tensor3dXf mix);
    int valid_length(int length);
    bool load_from_jit_module(const torch::jit::script::Module &module);
    void load_to_file(std::ofstream &outputstream);
    void load_from_file(std::ifstream &inputstream);
    float MaxAbsDifference(const DemucsModel &other) const;
    bool IsEqual(const DemucsModel &other, float tolerance = 1e-5)const;
    ~DemucsModel() {}
    std::vector<OneDecoder> decoders;
    std::vector<OneEncoder> encoders;
    LstmState lstm_state1, lstm_state2;
    OneLSTM lstm1, lstm2;
    int hidden, ch_scale, kernel_size, stride, chout, depth, resample;
    int lstm_hidden, chin;
    float floor;
    std::vector<Tensor3dXf> skips;
    int length;
    float std_constant;
    DemucsModel(int hidden = 48, int ch_scale = 2, int kernel_size = 8,
                int stride = 4, int chout = 1, int depth = 5, int chin = 1,
                int max_hidden = 10000, int growth = 2, int resample = 4,
                float floor = 1e-3);
};
struct DemucsStreamer {
    Tensor2dXf forward(Tensor2dXf wav);
    int stride;
    int numThreads;
    std::vector<DemucsModel> demucs_models;
    DemucsStreamer(int stride = 4096, int numThreads = 10);
};