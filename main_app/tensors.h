#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "Eigen/Dense"
#include <complex>
#include <unsupported/Eigen/CXX11/Tensor>

namespace Tensors {
typedef Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor> Tensor3dXh;
typedef Eigen::Tensor<std::complex<Eigen::half>, 3, Eigen::RowMajor> Tensor3dXch;
typedef Eigen::Tensor<Eigen::half, 1, Eigen::RowMajor> Tensor1dXh;
typedef Eigen::Tensor<Eigen::half, 4, Eigen::RowMajor> Tensor4dXh;
typedef Eigen::Vector<Eigen::half, Eigen::Dynamic> VectorXh;
typedef Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXh;
typedef Eigen::Tensor<float, 5> Tensor5dXf;
typedef Eigen::Tensor<float, 4> Tensor4dXf;
typedef Eigen::Tensor<float, 3> Tensor3dXf;
typedef Eigen::Tensor<float, 2> Tensor2dXf;
typedef Eigen::Tensor<float, 1> Tensor1dXf;
typedef Eigen::Tensor<std::complex<float>, 3> Tensor3dXcf;
} // namespace Eigen

#endif
