#include <torch/script.h> 
#include <iostream>
#include <vector>

void print_model_layers(const torch::jit::script::Module& module);