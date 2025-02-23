#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "Eigen/Dense"
#include <complex>
#include <iostream>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <variant>
#include <vector>

namespace Eigen {
typedef Tensor<Eigen::half, 3, Eigen::RowMajor> Tensor3dXh;
typedef Tensor<std::complex<Eigen::half>, 3, Eigen::RowMajor> Tensor3dXch;
typedef Tensor<Eigen::half, 1, Eigen::RowMajor> Tensor1dXh;
typedef Tensor<Eigen::half, 4, Eigen::RowMajor> Tensor4dXh;
typedef Vector<Eigen::half, Dynamic> VectorXh;

typedef Matrix<Eigen::half, Dynamic, Dynamic, Eigen::RowMajor> MatrixXh;

typedef Tensor<float, 4> Tensor4dXf;
typedef Tensor<float, 3> Tensor3dXf;
typedef Tensor<float, 2> Tensor2dXf;
typedef Tensor<float, 1> Tensor1dXf;
typedef Tensor<std::complex<float>, 3> Tensor3dXcf;
typedef std::variant<Tensor3dXh, Tensor3dXch, Tensor1dXh, Tensor4dXh, VectorXh,
                     MatrixXh, Tensor4dXf, Tensor3dXf, Tensor2dXf, Tensor1dXf,
                     Tensor3dXcf>
    VariantTensor; // v
} // namespace Eigen

#endif // TENSOR_HPP
