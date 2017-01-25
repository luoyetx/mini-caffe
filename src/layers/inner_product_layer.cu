#include <vector>

#include "./inner_product_layer.hpp"
#include "../filler.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void InnerProductLayer::Forward_gpu(const vector<Blob*>& bottom,
                                    const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  const real_t* weight = this->blobs_[0]->gpu_data();
  if (M_ == 1) {
    caffe_gpu_gemv(CblasNoTrans, N_, K_, static_cast<real_t>(1),
                   weight, bottom_data, static_cast<real_t>(0), top_data);
    if (bias_term_)
      caffe_gpu_axpy(N_, bias_multiplier_.cpu_data()[0],
                     this->blobs_[1]->gpu_data(), top_data);
  } else {
    caffe_gpu_gemm(CblasNoTrans,
                   transpose_ ? CblasNoTrans : CblasTrans,
                   M_, N_, K_, static_cast<real_t>(1),
                   bottom_data, weight, static_cast<real_t>(0), top_data);
    if (bias_term_)
      caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, M_, N_, 1,
                     static_cast<real_t>(1), bias_multiplier_.gpu_data(),
                     this->blobs_[1]->gpu_data(), static_cast<real_t>(1), top_data);
  }
}

}  // namespace caffe
