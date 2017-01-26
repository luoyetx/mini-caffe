#include <vector>

#include "./bias_layer.hpp"
#include "../filler.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

__global__ void BiasForward(const int n, const real_t* in,
                            const real_t* bias, const int bias_dim,
                            const int inner_dim, real_t* out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int bias_index = (index / inner_dim) % bias_dim;
    out[index] = in[index] + bias[bias_index];
  }
}

void BiasLayer::Forward_gpu(const vector<Blob*>& bottom,
                            const vector<Blob*>& top) {
  const int count = top[0]->count();
  const real_t* bottom_data = bottom[0]->gpu_data();
  const real_t* bias_data =
      ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  BiasForward  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bias_data, bias_dim_, inner_dim_, top_data);
}

}  // namespace caffe
