#ifdef USE_CUDNN

#include <vector>

#include "./cudnn_bn_layer.hpp"

namespace caffe {

void CuDNNBNLayer::LayerSetUp(const vector<Blob*>& bottom,
                              const vector<Blob*>& top) {
  BNLayer::LayerSetUp(bottom, top);
  if (this->bn_eps_ < CUDNN_BN_MIN_EPSILON) {
    LOG(WARNING) << "bn_eps is set to CUDNN_BN_MIN_EPSILON.";
    // Merely setting as CUDNN_BN_MIN_EPSILON fails the check due to
    // float / double precision problem.
    this->bn_eps_ = CUDNN_BN_MIN_EPSILON * 1.001;
  }
  scale_buf_.ReshapeLike(*(this->blobs_[0]));
  bias_buf_.ReshapeLike(*(this->blobs_[1]));
  save_mean_.ReshapeLike(*(this->blobs_[2]));
  save_inv_variance_.ReshapeLike(*(this->blobs_[3]));

  // Initialize CUDNN.
  CUDNN_CHECK(cudnnCreate(&handle_));
  cudnn::createTensor4dDesc(&bottom_desc_);
  cudnn::createTensor4dDesc(&top_desc_);
  cudnn::createTensor4dDesc(&bn_param_desc_);
  handles_setup_ = true;
}

void CuDNNBNLayer::Reshape(const vector<Blob*>& bottom,
                           const vector<Blob*>& top) {
  // Do not call BNLayer::Reshape function as some members are unnecessary
  this->num_ = bottom[0]->num();
  this->channels_ = bottom[0]->channels();
  this->height_ = bottom[0]->height();
  this->width_ = bottom[0]->width();

  top[0]->ReshapeLike(*(bottom[0]));

  // CUDNN tensors
  cudnn::setTensor4dDesc(&bottom_desc_, this->num_, this->channels_,
                                this->height_, this->width_);
  cudnn::setTensor4dDesc(&top_desc_, this->num_, this->channels_,
                                this->height_, this->width_);
  // Fix to the spatial mode
  CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bn_param_desc_,
      bottom_desc_, CUDNN_BATCHNORM_SPATIAL));
}

CuDNNBNLayer::~CuDNNBNLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);
  cudnnDestroyTensorDescriptor(bn_param_desc_);
  cudnnDestroy(handle_);
}

}  // namespace caffe

#endif  // USE_CUDNN
