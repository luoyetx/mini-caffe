#ifdef USE_CUDNN

#include <vector>

#include "./cudnn_pooling_layer.hpp"

namespace caffe {

void CuDNNPoolingLayer::LayerSetUp(const vector<Blob*>& bottom,
                                   const vector<Blob*>& top) {
  PoolingLayer::LayerSetUp(bottom, top);
  CUDNN_CHECK(cudnnCreate(&handle_));
  cudnn::createTensor4dDesc(&bottom_desc_);
  cudnn::createTensor4dDesc(&top_desc_);
  cudnn::createPoolingDesc(&pooling_desc_,
      this->layer_param_.pooling_param().pool(), &mode_,
      this->kernel_h_, this->kernel_w_, this->pad_h_, this->pad_w_,
      this->stride_h_, this->stride_w_);
  handles_setup_ = true;
}

void CuDNNPoolingLayer::Reshape(const vector<Blob*>& bottom,
                                const vector<Blob*>& top) {
  PoolingLayer::Reshape(bottom, top);
  cudnn::setTensor4dDesc(&bottom_desc_, bottom[0]->num(),
      this->channels_, this->height_, this->width_);
  cudnn::setTensor4dDesc(&top_desc_, bottom[0]->num(),
      this->channels_, this->pooled_height_, this->pooled_width_);
}

CuDNNPoolingLayer::~CuDNNPoolingLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);
  cudnnDestroyPoolingDescriptor(pooling_desc_);
  cudnnDestroy(handle_);
}

}   // namespace caffe

#endif  // USE_CUDNN
