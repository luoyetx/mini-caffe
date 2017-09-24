#include <vector>

#include "./lrn_layer.hpp"
#include "../util/math_functions.hpp"

#ifdef USE_CUDNN
#include "./cudnn/cudnn_lcn_layer.hpp"
#include "./cudnn/cudnn_lrn_layer.hpp"
#endif  // USE_CUDNN

namespace caffe {

void LRNLayer::LayerSetUp(const vector<Blob*>& bottom,
                          const vector<Blob*>& top) {
  size_ = this->layer_param_.lrn_param().local_size();
  CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for local_size";
  pre_pad_ = (size_ - 1) / 2;
  alpha_ = this->layer_param_.lrn_param().alpha();
  beta_ = this->layer_param_.lrn_param().beta();
  k_ = this->layer_param_.lrn_param().k();
  if (this->layer_param_.lrn_param().norm_region() ==
      LRNParameter_NormRegion_WITHIN_CHANNEL) {
    // Set up split_layer_ to use inputs in the numerator and denominator.
    split_top_vec_.clear();
    split_top_vec_.push_back(&product_input_);
    split_top_vec_.push_back(&square_input_);
    LayerParameter split_param;
    split_layer_.reset(new SplitLayer(split_param));
    split_layer_->SetUp(bottom, split_top_vec_);
    // Set up square_layer_ to square the inputs.
    square_bottom_vec_.clear();
    square_top_vec_.clear();
    square_bottom_vec_.push_back(&square_input_);
    square_top_vec_.push_back(&square_output_);
    LayerParameter square_param;
    square_param.mutable_power_param()->set_power(static_cast<real_t>(2));
    square_layer_.reset(new PowerLayer(square_param));
    square_layer_->SetUp(square_bottom_vec_, square_top_vec_);
    // Set up pool_layer_ to sum over square neighborhoods of the input.
    pool_top_vec_.clear();
    pool_top_vec_.push_back(&pool_output_);
    LayerParameter pool_param;
    pool_param.mutable_pooling_param()->set_pool(
        PoolingParameter_PoolMethod_AVE);
    pool_param.mutable_pooling_param()->set_pad(pre_pad_);
    pool_param.mutable_pooling_param()->set_kernel_size(size_);
    pool_layer_.reset(new PoolingLayer(pool_param));
    pool_layer_->SetUp(square_top_vec_, pool_top_vec_);
    // Set up power_layer_ to compute (1 + alpha_/N^2 s)^-beta_, where s is
    // the sum of a squared neighborhood (the output of pool_layer_).
    power_top_vec_.clear();
    power_top_vec_.push_back(&power_output_);
    LayerParameter power_param;
    power_param.mutable_power_param()->set_power(-beta_);
    power_param.mutable_power_param()->set_scale(alpha_);
    power_param.mutable_power_param()->set_shift(static_cast<real_t>(1));
    power_layer_.reset(new PowerLayer(power_param));
    power_layer_->SetUp(pool_top_vec_, power_top_vec_);
    // Set up a product_layer_ to compute outputs by multiplying inputs by the
    // inverse demoninator computed by the power layer.
    product_bottom_vec_.clear();
    product_bottom_vec_.push_back(&product_input_);
    product_bottom_vec_.push_back(&power_output_);
    LayerParameter product_param;
    EltwiseParameter* eltwise_param = product_param.mutable_eltwise_param();
    eltwise_param->set_operation(EltwiseParameter_EltwiseOp_PROD);
    product_layer_.reset(new EltwiseLayer(product_param));
    product_layer_->SetUp(product_bottom_vec_, top);
  }
}

void LRNLayer::Reshape(const vector<Blob*>& bottom,
                       const vector<Blob*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    top[0]->Reshape(num_, channels_, height_, width_);
    scale_.Reshape(num_, channels_, height_, width_);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    split_layer_->Reshape(bottom, split_top_vec_);
    square_layer_->Reshape(square_bottom_vec_, square_top_vec_);
    pool_layer_->Reshape(square_top_vec_, pool_top_vec_);
    power_layer_->Reshape(pool_top_vec_, power_top_vec_);
    product_layer_->Reshape(product_bottom_vec_, top);
    break;
  }
}

void LRNLayer::Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelForward_cpu(bottom, top);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    WithinChannelForward(bottom, top);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

void LRNLayer::CrossChannelForward_cpu(const vector<Blob*>& bottom,
                                       const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->cpu_data();
  real_t* top_data = top[0]->mutable_cpu_data();
  real_t* scale_data = scale_.mutable_cpu_data();
  // start with the constant value
  for (int i = 0; i < scale_.count(); ++i) {
    scale_data[i] = k_;
  }
  Blob padded_square(1, channels_ + size_ - 1, height_, width_);
  real_t* padded_square_data = padded_square.mutable_cpu_data();
  caffe_set(padded_square.count(), static_cast<real_t>(0), padded_square_data);
  real_t alpha_over_size = alpha_ / size_;
  // go through the images
  for (int n = 0; n < num_; ++n) {
    // compute the padded square
    caffe_sqr(channels_ * height_ * width_,
        bottom_data + bottom[0]->offset(n),
        padded_square_data + padded_square.offset(0, pre_pad_));
    // Create the first channel scale
    for (int c = 0; c < size_; ++c) {
      caffe_axpy(height_ * width_, alpha_over_size,
        padded_square_data + padded_square.offset(0, c),
        scale_data + scale_.offset(n, 0));
    }
    for (int c = 1; c < channels_; ++c) {
      // copy previous scale
      caffe_copy(height_ * width_,
        scale_data + scale_.offset(n, c - 1),
        scale_data + scale_.offset(n, c));
      // add head
      caffe_axpy(height_ * width_, alpha_over_size,
        padded_square_data + padded_square.offset(0, c + size_ - 1),
        scale_data + scale_.offset(n, c));
      // subtract tail
      caffe_axpy(height_ * width_, -alpha_over_size,
        padded_square_data + padded_square.offset(0, c - 1),
        scale_data + scale_.offset(n, c));
    }
  }

  // In the end, compute output
  caffe_powx(scale_.count(), scale_data, -beta_, top_data);
  caffe_mul(scale_.count(), top_data, bottom_data, top_data);
}

void LRNLayer::WithinChannelForward(const vector<Blob*>& bottom,
                                    const vector<Blob*>& top) {
  split_layer_->Forward(bottom, split_top_vec_);
  square_layer_->Forward(square_bottom_vec_, square_top_vec_);
  pool_layer_->Forward(square_top_vec_, pool_top_vec_);
  power_layer_->Forward(pool_top_vec_, power_top_vec_);
  product_layer_->Forward(product_bottom_vec_, top);
}

#ifndef USE_CUDA
STUB_GPU(LRNLayer);
STUB_GPU_FORWARD(LRNLayer, CrossChannelForward);
#endif

// Creator

// Get LRN layer according to engine
static shared_ptr<Layer> CreateLayer(const LayerParameter& param) {
#ifdef USE_CUDNN
  if (Caffe::mode() == Caffe::GPU) {
    LRNParameter lrn_param = param.lrn_param();
    if (lrn_param.norm_region() == LRNParameter_NormRegion_WITHIN_CHANNEL) {
      return shared_ptr<Layer>(new CuDNNLCNLayer(param));
    }
    else {
      // local size is too big to be handled through cuDNN
      if (param.lrn_param().local_size() > CUDNN_LRN_MAX_N) {
        return shared_ptr<Layer>(new LRNLayer(param));
      }
      else {
        return shared_ptr<Layer>(new CuDNNLRNLayer(param));
      }
    }
  }
#endif  // USE_CUDNN
  return shared_ptr<Layer>(new LRNLayer(param));
}

REGISTER_LAYER_CREATOR(LRN, CreateLayer);

}  // namespace caffe
