#pragma once
#include <torch/script.h> 
#include <iostream>
#include <vector>
#include "tensors.h"
void print_model_layers(const torch::jit::script::Module& module);
Eigen::Tensor1dXf load_vector(const torch::jit::script::Module& module);
void print_tensor(const torch::Tensor& tensor);