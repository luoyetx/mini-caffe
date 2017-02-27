#include <algorithm>
#include <vector>

#include "./prelu_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

// CUDA kernele for forward
__global__ void PReLUForward(const int n, const int channels, const int dim,
    const real_t* in, real_t* out, const real_t* slope_data,
    const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];
  }
}

void PReLULayer::Forward_gpu(const vector<Blob*>& bottom,
                             const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const real_t* slope_data = this->blobs_[0]->gpu_data();
  const int div_factor = channel_shared_ ? channels : 1;

  // NOLINT_NEXT_LINE(whitespace/operators)
  PReLUForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, channels, dim, bottom_data, top_data, slope_data, div_factor);
  CUDA_POST_KERNEL_CHECK;
}

}  // namespace caffe
