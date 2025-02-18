#include "layers.h"
#include "load_model.h"
#include "denoiser.h"
#include "tensors.h"
using namespace Eigen;
void TestSimpleModel();
void print_tensor(const torch::Tensor& tensor);
void print_model_layers(const torch::jit::script::Module& module);
Tensor1dXf TorchToEigen(const torch::Tensor& tensor);
// void TestWhoIsFasterTorchOrEigen();