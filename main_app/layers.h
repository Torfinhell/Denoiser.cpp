#pragma once
#include"tensors.h"
#include "filesystem"
#include "tensors.h"
#include <ATen/core/TensorBody.h>
#include <fstream>
#include <torch/script.h> 
namespace fs = std::filesystem;
using namespace Eigen;
struct Layer{
    public:
    virtual void forward(VariantTensor tensor){}
    virtual bool load_from_jit_module(torch::jit::script::Module module){return false;}
    virtual void load_to_file(std::ofstream& outputstream){}  // что делать с кодом возврата
    virtual void load_from_file(std::ifstream& inputstream){}
    virtual bool IsEqual(const Layer& other, float tolerance=1e-5){return false;}//checks if equal;used for testing 
//maybe write implementation for all tensors and just use it and maybe have a func that calls any func on all the tensors
    virtual float MaxAbsDifference(const Layer& other){return 0;}
    virtual ~Layer(){}
};
struct SimpleModel: public Layer{
    void forward(Tensor1dXf tensor);
    bool load_from_jit_module(torch::jit::script::Module module)override;
    void load_to_file(std::ofstream& outputstream) override;
    void load_from_file(std::ifstream& inputstream) override;
    float MaxAbsDifference(const SimpleModel& other);
    bool IsEqual(const SimpleModel& other, float tolerance=1e-5);
    ~SimpleModel(){}
    Tensor2dXf fc_weights;
    Tensor1dXf fc_bias;
    Tensor1dXf result;
};