#pragma once
#include <torch/script.h> 
#include <iostream>
#include <vector>

void load_tensor(const torch::jit::script::Module& module);