#include <vector>

#include "./embed_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

__global__ void EmbedForward(const int nthreads, const real_t* bottom_data,
    const real_t* weight, const int M, const int N, const int K,
    real_t* top_data) {
  CUDA_KERNEL_LOOP(top_index, nthreads) {
    const int n = top_index / N;
    const int d = top_index % N;
    const int index = static_cast<int>(bottom_data[n]);
    const int weight_index = index * N + d;
    top_data[top_index] = weight[weight_index];
  }
}

void EmbedLayer::Forward_gpu(const vector<Blob*>& bottom,
                             const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  const real_t* weight = this->blobs_[0]->gpu_data();
  const int count = top[0]->count();
  EmbedForward  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, weight, M_, N_, K_, top_data);
  if (bias_term_) {
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, M_, N_, 1, static_cast<real_t>(1),
      bias_multiplier_.gpu_data(),
      this->blobs_[1]->gpu_data(), static_cast<real_t>(1), top_data);
  }
}

}  // namespace caffe
