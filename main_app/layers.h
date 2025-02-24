#pragma once
#include "filesystem"
#include "tensors.h"
#include <ATen/core/TensorBody.h>
#include <fstream>
#include <torch/script.h>
extern Eigen::array<Eigen::IndexPair<int>, 1> product_dims_reg;
extern Eigen::array<Eigen::IndexPair<int>, 1> product_dims_sec_transposed;
extern Eigen::array<Eigen::IndexPair<int>, 1> product_dims_first_transposed;
extern Eigen::array<Eigen::IndexPair<int>, 1> product_dims_both_transposed;
extern std::array<long, 1> indices_dim_1;
extern std::array<long, 2> indices_dim_2;
extern std::array<long, 3> indices_dim_3;
extern std::array<long, 4> indices_dim_4;
namespace fs = std::filesystem;
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
    Tensor3dXf forward(Tensor3dXf tensor);
    bool load_from_jit_module(torch::jit::script::Module module) override;
    void load_to_file(std::ofstream &outputstream) override;
    void load_from_file(std::ifstream &inputstream) override;
    float MaxAbsDifference(const Conv1D &other);
    bool IsEqual(const Conv1D &other, float tolerance = 1e-5);
    ~Conv1D() {}
    int GetSize(int size, int kernel_size, int stride);
    int chin = 1, hidden = 2, kernel_size = 8, stride = 4; /////////////////
    Tensor3dXf conv_weights;
    Tensor1dXf conv_bias;
};
struct OneEncoder : public Layer {
    Tensor3dXf forward(Tensor3dXf tensor);
    bool load_from_jit_module(torch::jit::script::Module module) override;
    void load_to_file(std::ofstream &outputstream) override;
    void load_from_file(std::ifstream &inputstream) override;
    float MaxAbsDifference(const OneEncoder &other);
    bool IsEqual(const OneEncoder &other, float tolerance = 1e-5);
    ~OneEncoder() {}
    Conv1D convv1d;
};