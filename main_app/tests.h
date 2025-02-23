#include "load_model.h"
#include "denoiser.h"
#include "tensors.h"
using namespace Eigen;
void TestSimpleModel();
#include <iostream>
#include <array>
#include <Eigen/Dense>
#include <type_traits>


void print_model_layers(const torch::jit::script::Module& module);

void print_tensor(const torch::Tensor& tensor);

void print_1d_tensor(Tensor1dXf tensor);

void print_2d_tensor(Tensor2dXf tensor);

