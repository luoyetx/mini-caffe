#include <algorithm>
#include <vector>

#include "./bn_layer.hpp"
#include "../filler.hpp"
#include "../util/math_functions.hpp"

#ifdef USE_CUDNN
#include "./cudnn/cudnn_bn_layer.hpp"
#endif  // USE_CUDNN

namespace caffe {

void BNLayer::LayerSetUp(const vector<Blob*>& bottom,
                         const vector<Blob*>& top) {
  frozen_ = this->layer_param_.bn_param().frozen();
  bn_momentum_ = this->layer_param_.bn_param().momentum();
  bn_eps_ = this->layer_param_.bn_param().eps();
  // Initialize parameters
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(4);
    vector<int> shape;
    shape.push_back(1);
    shape.push_back(bottom[0]->channels());
    // slope
    this->blobs_[0].reset(new Blob(shape));
    shared_ptr<Filler> slope_filler(GetFiller(
        this->layer_param_.bn_param().slope_filler()));
    slope_filler->Fill(this->blobs_[0].get());
    // bias
    this->blobs_[1].reset(new Blob(shape));
    shared_ptr<Filler> bias_filler(GetFiller(
        this->layer_param_.bn_param().bias_filler()));
    bias_filler->Fill(this->blobs_[1].get());
    // moving average mean
    this->blobs_[2].reset(new Blob(shape));
    caffe_set(this->blobs_[2]->count(), static_cast<real_t>(0),
        this->blobs_[2]->mutable_cpu_data());
    // moving average variance
    this->blobs_[3].reset(new Blob(shape));
    caffe_set(this->blobs_[3]->count(), static_cast<real_t>(1),
        this->blobs_[3]->mutable_cpu_data());
  }
  // set temp blob name
  broadcast_buffer_.set_name(this->layer_param_.name() + "__broadcast_buffer__");
  spatial_statistic_.set_name(this->layer_param_.name() + "__spatial_statistic__");
  x_norm_.set_name(this->layer_param_.name() + "__x_norm__");
  spatial_sum_multiplier_.set_name(this->layer_param_.name() + "__spatial_sum_multiplier__");
}

void BNLayer::Reshape(const vector<Blob*>& bottom,
                      const vector<Blob*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  top[0]->ReshapeLike(*(bottom[0]));

  broadcast_buffer_.ReshapeLike(*(bottom[0]));
  spatial_statistic_.Reshape(num_, channels_, 1, 1);
  batch_statistic_.Reshape(1, channels_, 1, 1);

  x_norm_.ReshapeLike(*(bottom[0]));
  x_inv_std_.ReshapeLike(batch_statistic_);

  spatial_sum_multiplier_.Reshape(1, 1, height_, width_);
  caffe_set(spatial_sum_multiplier_.count(), static_cast<real_t>(1),
      spatial_sum_multiplier_.mutable_cpu_data());
  batch_sum_multiplier_.Reshape(num_, 1, 1, 1);
  caffe_set(batch_sum_multiplier_.count(), static_cast<real_t>(1),
      batch_sum_multiplier_.mutable_cpu_data());
}

void BNLayer::Forward_cpu(const vector<Blob*>& bottom,
                          const vector<Blob*>& top) {
  const real_t* const_bottom_data = bottom[0]->cpu_data();
  const real_t* const_top_data = top[0]->cpu_data();
  real_t* top_data = top[0]->mutable_cpu_data();

  const real_t* scale_data = this->blobs_[0]->cpu_data();
  const real_t* shift_data = this->blobs_[1]->cpu_data();

  // Mean normalization
  // Use the moving average mean
  caffe_copy(batch_statistic_.count(), this->blobs_[2]->cpu_data(),
      batch_statistic_.mutable_cpu_data());
  // Broadcast the mean vector
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      static_cast<real_t>(1), batch_sum_multiplier_.cpu_data(), batch_statistic_.cpu_data(),
      static_cast<real_t>(0), spatial_statistic_.mutable_cpu_data());
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, static_cast<real_t>(-1),
      spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(),
      static_cast<real_t>(0), broadcast_buffer_.mutable_cpu_data());
  // Subtract
  caffe_add(broadcast_buffer_.count(), const_bottom_data,
      broadcast_buffer_.cpu_data(), top_data);

  // Variance normalization
  // Use the moving average variance
  caffe_copy(batch_statistic_.count(), this->blobs_[3]->cpu_data(),
      batch_statistic_.mutable_cpu_data());
  // Broadcast the inverse std
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
        static_cast<real_t>(1), batch_sum_multiplier_.cpu_data(), batch_statistic_.cpu_data(),
        static_cast<real_t>(0), spatial_statistic_.mutable_cpu_data());
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, static_cast<real_t>(1),
      spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(),
      static_cast<real_t>(0), broadcast_buffer_.mutable_cpu_data());
  // Multiply with the inverse std
  caffe_mul(broadcast_buffer_.count(), const_top_data,
      broadcast_buffer_.cpu_data(), top_data);

  // Save the normalized inputs and std for backprop
  if (!frozen_) {
    caffe_copy(broadcast_buffer_.count(), const_top_data,
        x_norm_.mutable_cpu_data());
    caffe_copy(batch_statistic_.count(), batch_statistic_.cpu_data(),
        x_inv_std_.mutable_cpu_data());
  }

  // Scale
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      static_cast<real_t>(1), batch_sum_multiplier_.cpu_data(), scale_data,
      static_cast<real_t>(0), spatial_statistic_.mutable_cpu_data());
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, static_cast<real_t>(1),
      spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(),
      static_cast<real_t>(0), broadcast_buffer_.mutable_cpu_data());
  caffe_mul(broadcast_buffer_.count(), const_top_data,
      broadcast_buffer_.cpu_data(), top_data);

  // Shift
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      static_cast<real_t>(1), batch_sum_multiplier_.cpu_data(), shift_data,
      static_cast<real_t>(0), spatial_statistic_.mutable_cpu_data());
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, static_cast<real_t>(1),
      spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(),
      static_cast<real_t>(0), broadcast_buffer_.mutable_cpu_data());
  caffe_add(broadcast_buffer_.count(), const_top_data,
      broadcast_buffer_.cpu_data(), top_data);
}

#ifndef USE_CUDA
STUB_GPU(BNLayer);
#endif

// Creator

static shared_ptr<Layer> CreateLayer(const LayerParameter &param) {
#ifdef USE_CUDNN
  if (Caffe::mode() == Caffe::GPU) {
    return shared_ptr<Layer>(new CuDNNBNLayer(param));
  }
#endif  // USE_CUDNN
  return shared_ptr<Layer>(new BNLayer(param));
}

REGISTER_LAYER_CREATOR(BN, CreateLayer);

}  // namespace caffe
