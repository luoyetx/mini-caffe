#include <algorithm>
#include <vector>

#include "./bn_layer.hpp"
#include "../filler.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void BNLayer::Forward_gpu(const vector<Blob*>& bottom,
                          const vector<Blob*>& top) {
  const real_t* const_bottom_data = bottom[0]->gpu_data();
  const real_t* const_top_data = top[0]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();

  const real_t* scale_data = this->blobs_[0]->gpu_data();
  const real_t* shift_data = this->blobs_[1]->gpu_data();

  // Mean normalization
  // Use the moving average mean
  caffe_copy(batch_statistic_.count(), this->blobs_[2]->gpu_data(),
      batch_statistic_.mutable_gpu_data());
  // Broadcast the mean vector
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      static_cast<real_t>(1), batch_sum_multiplier_.gpu_data(), batch_statistic_.gpu_data(),
      static_cast<real_t>(0), spatial_statistic_.mutable_gpu_data());
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, static_cast<real_t>(-1),
      spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
      static_cast<real_t>(0), broadcast_buffer_.mutable_gpu_data());
  // Subtract
  caffe_gpu_add(broadcast_buffer_.count(), const_bottom_data,
      broadcast_buffer_.gpu_data(), top_data);

  // Variance normalization
  // Use the moving average variance
  caffe_copy(batch_statistic_.count(), this->blobs_[3]->gpu_data(),
      batch_statistic_.mutable_gpu_data());
  // Broadcast the inverse std
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      static_cast<real_t>(1), batch_sum_multiplier_.gpu_data(), batch_statistic_.gpu_data(),
      static_cast<real_t>(0), spatial_statistic_.mutable_gpu_data());
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, static_cast<real_t>(1),
      spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
      static_cast<real_t>(0), broadcast_buffer_.mutable_gpu_data());
  // Multiply with the inverse std
  caffe_gpu_mul(broadcast_buffer_.count(), const_top_data,
      broadcast_buffer_.gpu_data(), top_data);

  // Save the normalized inputs and std for backprop
  if (!frozen_) {
    caffe_copy(broadcast_buffer_.count(), const_top_data,
        x_norm_.mutable_gpu_data());
    caffe_copy(batch_statistic_.count(), batch_statistic_.gpu_data(),
        x_inv_std_.mutable_gpu_data());
  }

  // Scale
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      static_cast<real_t>(1), batch_sum_multiplier_.gpu_data(), scale_data,
      static_cast<real_t>(0), spatial_statistic_.mutable_gpu_data());
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, static_cast<real_t>(1),
      spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
      static_cast<real_t>(0), broadcast_buffer_.mutable_gpu_data());
  caffe_gpu_mul(broadcast_buffer_.count(), const_top_data,
      broadcast_buffer_.gpu_data(), top_data);

  // Shift
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      static_cast<real_t>(1), batch_sum_multiplier_.gpu_data(), shift_data,
      static_cast<real_t>(0), spatial_statistic_.mutable_gpu_data());
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, static_cast<real_t>(1),
      spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
      static_cast<real_t>(0), broadcast_buffer_.mutable_gpu_data());
  caffe_gpu_add(broadcast_buffer_.count(), const_top_data,
      broadcast_buffer_.gpu_data(), top_data);
}

}  // namespace caffe
