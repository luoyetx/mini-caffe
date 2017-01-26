#include <vector>

#include "./conv_layer.hpp"

namespace caffe {

void ConvolutionLayer::Forward_gpu(const vector<Blob*>& bottom,
                                   const vector<Blob*>& top) {
  const real_t* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const real_t* bottom_data = bottom[i]->gpu_data();
    real_t* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const real_t* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

}  // namespace caffe
