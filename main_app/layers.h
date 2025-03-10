#pragma once
#include "filesystem"
#include "tensors.h"
#include <ATen/core/TensorBody.h>
#include <fstream>
#include <torch/script.h>
#include <vector>
extern Eigen::array<Eigen::IndexPair<int>, 1> product_dims_reg;
extern Eigen::array<Eigen::IndexPair<int>, 1> product_dims_sec_transposed;
extern Eigen::array<Eigen::IndexPair<int>, 1> product_dims_first_transposed;
extern Eigen::array<Eigen::IndexPair<int>, 1> product_dims_both_transposed;
extern std::array<long, 1> indices_dim_1;
extern std::array<long, 2> indices_dim_2;
extern std::array<long, 3> indices_dim_3;
extern std::array<long, 4> indices_dim_4;
using namespace Eigen;
struct Layer {
  public:
    virtual bool load_from_jit_module(torch::jit::script::Module module)
    {
        return false;
    }
    virtual void load_to_file(std::ofstream &outputstream) {}
    virtual void load_from_file(std::ifstream &inputstream) {}
    virtual bool IsEqual(const Layer &other, float tolerance = 1e-5)
    {
        return false;
    }
    virtual float MaxAbsDifference(const Layer &other) { return 0; }
    virtual ~Layer() {}
};
struct SimpleModel : public Layer {
    void forward(Tensor1dXf tensor);
    bool load_from_jit_module(torch::jit::script::Module module) override;
    void load_to_file(std::ofstream &outputstream) override;
    void load_from_file(std::ifstream &inputstream) override;
    float MaxAbsDifference(const SimpleModel &other);
    bool IsEqual(const SimpleModel &other, float tolerance = 1e-5);
    ~SimpleModel() {}

    Tensor2dXf fc_weights;
    Tensor1dXf fc_bias;
    Tensor1dXf result;
};
struct Conv1D : public Layer {
    Tensor3dXf forward(Tensor3dXf tensor, int InputChannels, int OutputChannels,
                       int kernel_size = 3, int stride = 1, int padding = 0);
    bool load_from_jit_module(torch::jit::script::Module module,
                              std::string weights_index);
    void load_to_file(std::ofstream &outputstream) override;
    void load_from_file(std::ifstream &inputstream) override;
    float MaxAbsDifference(const Conv1D &other);
    bool IsEqual(const Conv1D &other, float tolerance = 1e-5);
    ~Conv1D() {}
    int GetSize(int size, int kernel_size, int stride);
    Tensor3dXf conv_weights;
    Tensor1dXf conv_bias;
};
struct ConvTranspose1d : public Layer {
    Tensor3dXf forward(Tensor3dXf tensor, int InputChannels, int OutputChannels,
                       int kernel_size = 3, int stride = 1);
    bool load_from_jit_module(torch::jit::script::Module module,
                              std::string weights_index);
    void load_to_file(std::ofstream &outputstream) override;
    void load_from_file(std::ifstream &inputstream) override;
    float MaxAbsDifference(const ConvTranspose1d &other);
    bool IsEqual(const ConvTranspose1d &other, float tolerance = 1e-5);
    ~ConvTranspose1d() {}
    int GetTransposedSize(int size, int kernel_size, int stride);
    Tensor3dXf conv_tr_weights;
    Tensor1dXf conv_tr_bias;
};
struct RELU : public Layer {
    Tensor3dXf forward(Tensor3dXf tensor);
};
struct GLU : public Layer {
    Tensor3dXf forward(Tensor3dXf tensor);
};
struct OneEncoder : public Layer {
    Tensor3dXf forward(Tensor3dXf tensor);
    bool load_from_jit_module(torch::jit::script::Module module,
                              std::string weights_index = "0");
    void load_to_file(std::ofstream &outputstream) override;
    void load_from_file(std::ifstream &inputstream) override;
    float MaxAbsDifference(const OneEncoder &other);
    bool IsEqual(const OneEncoder &other, float tolerance = 1e-5);
    ~OneEncoder() {}
    Conv1D conv_1_1d;
    Conv1D conv_2_1d;
    RELU relu;
    GLU glu;
    int hidden, ch_scale, kernel_size, stride, chout, chin;
    OneEncoder(int hidden = 48, int ch_scale = 2, int kernel_size = 8,
               int stride = 4, int chout = 1, int chin = 1)
        : hidden(hidden), ch_scale(ch_scale), kernel_size(kernel_size),
          stride(stride), chout(chout), chin(chin)
    {
    }
};

struct OneDecoder : public Layer {
    Tensor3dXf forward(Tensor3dXf tensor);
    bool load_from_jit_module(torch::jit::script::Module module,
                              std::string weights_index = "0");
    void load_to_file(std::ofstream &outputstream) override;
    void load_from_file(std::ifstream &inputstream) override;
    float MaxAbsDifference(const OneDecoder &other);
    bool IsEqual(const OneDecoder &other, float tolerance = 1e-5);
    ~OneDecoder() {}
    Conv1D conv_1_1d;
    GLU glu;
    ConvTranspose1d conv_tr_1_1d;
    int hidden, ch_scale, kernel_size, stride, chout;
    OneDecoder(int hidden = 48, int ch_scale = 2, int kernel_size = 8,
               int stride = 4, int chout = 1)
        : hidden(hidden), ch_scale(ch_scale), kernel_size(kernel_size),
          stride(stride), chout(chout)
    {
    }
};
struct OneLSTM : public Layer {
    Tensor3dXf forward(Tensor3dXf tensor, int HiddenSize, bool bi = false);
    bool load_from_jit_module(torch::jit::script::Module module,
                              std::string weights_index);
    void load_to_file(std::ofstream &outputstream) override;
    void load_from_file(std::ifstream &inputstream) override;
    float MaxAbsDifference(const OneLSTM &other);
    bool IsEqual(const OneLSTM &other, float tolerance = 1e-5);
    ~OneLSTM() {}
    Tensor2dXf lstm_weight_ih, lstm_weight_hh;
    Tensor2dXf lstm_bias_ih, lstm_bias_hh;
};

struct SimpleEncoderDecoder : public Layer {
    Tensor3dXf forward(Tensor3dXf tensor);
    bool load_from_jit_module(torch::jit::script::Module module) override;
    void load_to_file(std::ofstream &outputstream) override;
    void load_from_file(std::ifstream &inputstream) override;
    float MaxAbsDifference(const SimpleEncoderDecoder &other);
    bool IsEqual(const SimpleEncoderDecoder &other, float tolerance = 1e-5);
    ~SimpleEncoderDecoder() {}
    OneEncoder one_encoder;
    OneDecoder one_decoder;
    int hidden, ch_scale, kernel_size, stride, chout;
    SimpleEncoderDecoder(int hidden = 48, int ch_scale = 2, int kernel_size = 8,
                         int stride = 4, int chout = 1)
        : hidden(hidden), ch_scale(ch_scale), kernel_size(kernel_size),
          stride(stride), chout(chout),
          one_encoder(hidden, ch_scale, kernel_size, stride),
          one_decoder(hidden, ch_scale, kernel_size, stride)
    {
    }
};
struct SimpleEncoderDecoderLSTM : public Layer {
    Tensor3dXf forward(Tensor3dXf tensor);
    bool load_from_jit_module(torch::jit::script::Module module) override;
    void load_to_file(std::ofstream &outputstream) override;
    void load_from_file(std::ifstream &inputstream) override;
    float MaxAbsDifference(const SimpleEncoderDecoderLSTM &other);
    bool IsEqual(const SimpleEncoderDecoderLSTM &other, float tolerance = 1e-5);
    ~SimpleEncoderDecoderLSTM() {}
    OneEncoder one_encoder;
    OneDecoder one_decoder;
    OneLSTM lstm1, lstm2;
    int hidden, ch_scale, kernel_size, stride, chout;
    SimpleEncoderDecoderLSTM(int hidden = 48, int ch_scale = 2,
                             int kernel_size = 8, int stride = 4, int chout = 1)
        : hidden(hidden), ch_scale(ch_scale), kernel_size(kernel_size),
          stride(stride), chout(chout),
          one_encoder(hidden, ch_scale, kernel_size, stride),
          one_decoder(hidden, ch_scale, kernel_size, stride)
    {
    }
};
struct DemucsModel : public Layer {
    Tensor3dXf forward(Tensor3dXf tensor);
    int valid_length(int length);
    bool load_from_jit_module(torch::jit::script::Module module) override;
    void load_to_file(std::ofstream &outputstream) override;
    void load_from_file(std::ifstream &inputstream) override;
    float MaxAbsDifference(const DemucsModel &other);
    bool IsEqual(const DemucsModel &other, float tolerance = 1e-5);
    ~DemucsModel() {}
    std::vector<OneDecoder> decoders;
    std::vector<OneEncoder> encoders;
    OneLSTM lstm1, lstm2;
    int hidden, ch_scale, kernel_size, stride, chout, depth, resample;
    int lstm_hidden;
    DemucsModel(int hidden = 48, int ch_scale = 2, int kernel_size = 8,
                int stride = 4, int chout = 1, int depth = 5, int chin = 1,
                int max_hidden = 10000, int growth = 2, int resample =4)
        : hidden(hidden), ch_scale(ch_scale), kernel_size(kernel_size),
          stride(stride), chout(chout), depth(depth),resample(resample)
    {
        for (int i = 0; i < depth; i++) {
            encoders.emplace_back(hidden, ch_scale, kernel_size, stride, chout,
                                  chin);
            decoders.emplace_back(hidden, ch_scale, kernel_size, stride, chout);
            chout = hidden;
            chin = hidden;
            hidden = std::min(int(growth * hidden), max_hidden);
        }
        std::reverse(decoders.begin(), decoders.end());
        lstm_hidden = chin;
    }
};