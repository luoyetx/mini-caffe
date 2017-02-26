#include <cfloat>
#include <vector>
#include <limits>

#include "./eltwise_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

__global__ void MaxForward(const int nthreads, const real_t* bottom_data_a,
                           const real_t* bottom_data_b, const int blob_idx,
                           real_t* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    top_data[index] = max(bottom_data_a[index], bottom_data_b[index]);
  }
}

void EltwiseLayer::Forward_gpu(const vector<Blob*>& bottom,
                               const vector<Blob*>& top) {
  const int count = top[0]->count();
  real_t* top_data = top[0]->mutable_gpu_data();
  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
    caffe_gpu_mul(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
        top_data);
    for (int i = 2; i < bottom.size(); ++i) {
      caffe_gpu_mul(count, top_data, bottom[i]->gpu_data(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_SUM:
    caffe_gpu_set(count, static_cast<real_t>(0), top_data);
    // TODO(shelhamer) does cuBLAS optimize to sum for coeff = 1?
    for (int i = 0; i < bottom.size(); ++i) {
      caffe_gpu_axpy(count, coeffs_[i], bottom[i]->gpu_data(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_MAX:
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 0, top_data);
    for (int i = 2; i < bottom.size(); ++i) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      MaxForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, top_data, bottom[i]->gpu_data(), i-1, top_data);
    }
    break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
}

}  // namespace caffe
