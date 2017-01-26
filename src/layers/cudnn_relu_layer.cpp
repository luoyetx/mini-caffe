#ifdef USE_CUDNN

#include <vector>

#include "./cudnn_relu_layer.hpp"

namespace caffe {

void CuDNNReLULayer::LayerSetUp(const vector<Blob*>& bottom,
                                const vector<Blob*>& top) {
  ReLULayer::LayerSetUp(bottom, top);
  // initialize cuDNN
  CUDNN_CHECK(cudnnCreate(&handle_));
  cudnn::createTensor4dDesc(&bottom_desc_);
  cudnn::createTensor4dDesc(&top_desc_);
  cudnn::createActivationDescriptor(&activ_desc_, CUDNN_ACTIVATION_RELU);
  handles_setup_ = true;
}

void CuDNNReLULayer::Reshape(const vector<Blob*>& bottom,
                             const vector<Blob*>& top) {
  ReLULayer::Reshape(bottom, top);
  const int N = bottom[0]->num();
  const int K = bottom[0]->channels();
  const int H = bottom[0]->height();
  const int W = bottom[0]->width();
  cudnn::setTensor4dDesc(&bottom_desc_, N, K, H, W);
  cudnn::setTensor4dDesc(&top_desc_, N, K, H, W);
}

CuDNNReLULayer::~CuDNNReLULayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(this->bottom_desc_);
  cudnnDestroyTensorDescriptor(this->top_desc_);
  cudnnDestroy(this->handle_);
}

}  // namespace caffe

#endif  // USE_CUDNN
