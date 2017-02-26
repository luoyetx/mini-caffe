#include <cfloat>
#include <vector>

#include "./scale_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

__global__ void ScaleForward(const int n, const real_t* in,
    const real_t* scale, const int scale_dim, const int inner_dim,
    real_t* out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int scale_index = (index / inner_dim) % scale_dim;
    out[index] = in[index] * scale[scale_index];
  }
}

__global__ void ScaleBiasForward(const int n, const real_t* in,
    const real_t* scale, const real_t* bias,
    const int scale_dim, const int inner_dim, real_t* out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int scale_index = (index / inner_dim) % scale_dim;
    out[index] = in[index] * scale[scale_index] + bias[scale_index];
  }
}

void ScaleLayer::Forward_gpu(const vector<Blob*>& bottom,
                             const vector<Blob*>& top) {
  const int count = top[0]->count();
  const real_t* bottom_data = bottom[0]->gpu_data();
  const real_t* scale_data =
      ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  if (bias_layer_) {
    const real_t* bias_data = this->blobs_[bias_param_id_]->gpu_data();
    ScaleBiasForward  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, scale_data, bias_data, scale_dim_, inner_dim_,
        top_data);
  } else {
    ScaleForward  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, scale_data, scale_dim_, inner_dim_, top_data);
  }
}

}  // namespace caffe
