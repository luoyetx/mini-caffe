#include <algorithm>
#include <vector>

#include "./batch_norm_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void BatchNormLayer::Forward_gpu(const vector<Blob*>& bottom,
                                 const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count()/(channels_*bottom[0]->shape(0));

  if (bottom[0] != top[0]) {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }

  if (use_global_stats_) {
    // use the stored mean/variance estimates.
    const real_t scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
        0 : 1 / this->blobs_[2]->cpu_data()[0];
    caffe_gpu_scale(variance_.count(), scale_factor,
        this->blobs_[0]->gpu_data(), mean_.mutable_gpu_data());
    caffe_gpu_scale(variance_.count(), scale_factor,
        this->blobs_[1]->gpu_data(), variance_.mutable_gpu_data());
  } else {
    // compute mean
    caffe_gpu_gemv(CblasNoTrans, channels_ * num, spatial_dim,
        1. / (num * spatial_dim), bottom_data,
        spatial_sum_multiplier_.gpu_data(), 0,
        num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemv(CblasTrans, num, channels_, 1,
        num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0,
        mean_.mutable_gpu_data());
  }

  // subtract mean
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), mean_.gpu_data(), 0,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, -1, num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 1, top_data);

  if (!use_global_stats_) {
    // compute variance using var(X) = E((X-EX)^2)
    caffe_gpu_powx(top[0]->count(), top_data, 2,
        temp_.mutable_gpu_data());  // (X-EX)^2
    caffe_gpu_gemv(CblasNoTrans, channels_ * num, spatial_dim,
        1. / (num * spatial_dim), temp_.gpu_data(),
        spatial_sum_multiplier_.gpu_data(), 0,
        num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemv(CblasTrans, num, channels_, 1,
        num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0,
        variance_.mutable_gpu_data());  // E((X_EX)^2)
  }

  // normalize variance
  caffe_gpu_add_scalar(variance_.count(), eps_, variance_.mutable_gpu_data());
  caffe_gpu_powx(variance_.count(), variance_.gpu_data(), static_cast<real_t>(0.5),
                 variance_.mutable_gpu_data());

  // replicate variance to input size
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), variance_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 0., temp_.mutable_gpu_data());
  caffe_gpu_div(temp_.count(), top_data, temp_.gpu_data(), top_data);
}

}  // namespace caffe
