#include "layers.h"
#include "tests.h"
Tensor3dXf KernelUpsample2(int zeros = 56);
template <typename Tensor, int NumDim>
void GetMean(const Tensor &tensor, Tensor &result, int dim,
             std::array<long, NumDim> arr, int pos = 0,
             bool is_first_call = true)
{
    if (is_first_call) {
        std::array<long, NumDim> dimensions = {};
        for (int dimension = 0; dimension < NumDim; dimension++) {
            dimensions[dimension] = tensor.dimension(dimension);
        }
        dimensions[dim] = 1;
        result.resize(dimensions);
        result.setZero();
    }

    if (pos == NumDim) {
        std::array<long, NumDim> arr_copy = arr;
        double sum = 0;
        for (int ind = 0; ind < tensor.dimension(dim); ind++) {
            arr[dim] = ind;
            sum += tensor(arr);
        }
        result(arr_copy) = sum / tensor.dimension(dim);
    }
    else if (pos == dim) {
        arr[dim] = 0;
        GetMean<Tensor, NumDim>(tensor, result, dim, arr, pos + 1, false);
    }
    else {
        for (int ind = 0; ind < tensor.dimension(pos); ind++) {
            arr[pos] = ind;
            GetMean<Tensor, NumDim>(tensor, result, dim, arr, pos + 1, false);
        }
    }
}
template <typename Tensor, int NumDim>
void GetStd(Tensor &tensor, Tensor &result, int dim,
            std::array<long, NumDim> arr, int pos = 0,
            bool is_first_call = true)
{
    if (is_first_call) {
        std::array<long, NumDim> dimensions = {};
        for (int dimension = 0; dimension < NumDim; dimension++) {
            dimensions[dimension] = tensor.dimension(dimension);
        }
        dimensions[dim] = 1;
        result.resize(dimensions);
        result.setZero();
    }

    if (pos == NumDim) {
        std::array<long, NumDim> arr_copy = arr;
        double mean = 0;
        long count = tensor.dimension(dim);
        for (int ind = 0; ind < count; ind++) {
            arr[dim] = ind;
            mean += tensor(arr);
        }
        mean /= count;
        for (int ind = 0; ind < count; ind++) {
            arr[dim] = ind;
            double diff = tensor(arr) - mean;
            result(arr_copy) += diff * diff;
        }
        if (count > 1) {
            result(arr_copy) /= (count - 1);
            result(arr_copy) = sqrt(result(arr_copy));
        }
        else {
            result(arr_copy) = 0;
        }
    }
    else if (pos == dim) {
        arr[dim] = 0;
        GetStd<Tensor, NumDim>(tensor, result, dim, arr, pos + 1, false);
    }
    else {
        for (int ind = 0; ind < tensor.dimension(pos); ind++) {
            arr[pos] = ind;
            GetStd<Tensor, NumDim>(tensor, result, dim, arr, pos + 1, false);
        }
    }
}
template <typename Tensor, typename TensorMore, int NumDim>
Tensor UpSample(Tensor tensor, int zeros = 56)
{
    int last_dimension = tensor.dimension(NumDim - 1);
    Tensor kernel = KernelUpsample2(zeros);
    Conv1D conv;
    conv.conv_weights = kernel;
    conv.conv_bias.resize(std::array<long, 1>{1});
    conv.conv_bias.setZero();
    Tensor3dXf res =
        conv.forward(tensor.reshape(std::array<long, 3>{
                         tensor.size() / last_dimension, 1, last_dimension}),
                     1, 1, kernel.dimension(2), 1, zeros);
    std::array<long, NumDim> extent;
    for (long i = 0; i < NumDim; i++) {
        extent[i] = res.dimension(i);
    }
    extent[NumDim - 1]--;
    std::array<long, NumDim> offset = {};
    offset[NumDim - 1] = 1;
    Tensor out = res.slice(offset, extent).reshape(tensor.dimensions());
    std::array<long, NumDim + 1> new_shape;
    for (int i = 0; i < NumDim; i++) {
        new_shape[i] = tensor.dimension(i);
    }
    new_shape[NumDim] = new_shape[NumDim - 1];
    new_shape[NumDim - 1] = 2;
    TensorMore stacked(new_shape);
    stacked.chip(0, NumDim - 1) = tensor;
    stacked.chip(1, NumDim - 1) = out;
    new_shape[NumDim - 1] = 2 * new_shape[NumDim];
    new_shape[NumDim] = 1;
    TensorMore stacked_reshaped = stacked.reshape(new_shape);
    Tensor y = stacked_reshaped.chip(0, NumDim);
    return y;
}

template <typename Tensor, typename TensorMore, int NumDim>
Tensor DownSample(Tensor tensor, int zeros = 56)
{
    Tensor3dXf padded_tensor = tensor;
    if (tensor.dimension(NumDim - 1) % 2 != 0) {
        Eigen::array<std::pair<int, int>, 3> paddings = {};
        paddings[2] = std::make_pair(0, 1);
        padded_tensor = tensor.pad(paddings);
    }
    std::array<long, NumDim> new_shape = padded_tensor.dimensions();
    new_shape[NumDim - 1] /= 2;

    Tensor3dXf tensor_even(new_shape), tensor_odd(new_shape);
    for (int i = 0; i < padded_tensor.dimension(NumDim - 1); i++) {
        if (i % 2 == 0) {
            tensor_even.chip(i / 2, NumDim - 1) =
                padded_tensor.chip(i, NumDim - 1);
        }
        else {
            tensor_odd.chip((i - 1) / 2, NumDim - 1) =
                padded_tensor.chip(i, NumDim - 1);
        }
    }
    long first_shape_size = 1,
         last_dimension = tensor_odd.dimension(NumDim - 1);
    for (long i = 0; i < NumDim - 1; i++) {
        first_shape_size *= tensor_odd.dimension(i);
    }
    Tensor kernel = KernelUpsample2(zeros);
    Conv1D conv;
    conv.conv_weights = kernel;
    conv.conv_bias.resize(std::array<long, 1>{1});
    conv.conv_bias.setZero();
    Tensor3dXf res = conv.forward(tensor_odd.reshape(std::array<long, 3>{
                                      first_shape_size, 1, last_dimension}),
                                  1, 1, kernel.dimension(2), 1, zeros);
    std::array<long, NumDim> extent = res.dimensions();
    extent[NumDim - 1]--;
    std::array<long, NumDim> offset = {};
    Tensor out = res.slice(offset, extent).reshape(tensor_odd.dimensions());
    return (tensor_even + out).unaryExpr([](float x) { return x / 2; });
}
