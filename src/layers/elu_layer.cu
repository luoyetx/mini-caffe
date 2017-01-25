#include <algorithm>
#include <vector>

#include "./elu_layer.hpp"

namespace caffe {

__global__ void ELUForward(const int n, const real_t* in, real_t* out,
                           real_t alpha) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] :
        alpha * (exp(in[index]) - 1);
  }
}

void ELULayer::Forward_gpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  real_t alpha = this->layer_param_.elu_param().alpha();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ELUForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, alpha);
  CUDA_POST_KERNEL_CHECK;
}

}  // namespace caffe
