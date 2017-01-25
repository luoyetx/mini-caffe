#ifdef USE_CUDNN

#include <vector>

#include "./cudnn_softmax_layer.hpp"

namespace caffe {

void CuDNNSoftmaxLayer::LayerSetUp(const vector<Blob*>& bottom,
                                   const vector<Blob*>& top) {
  SoftmaxLayer::LayerSetUp(bottom, top);
  // Initialize CUDNN.
  CUDNN_CHECK(cudnnCreate(&handle_));
  cudnn::createTensor4dDesc(&bottom_desc_);
  cudnn::createTensor4dDesc(&top_desc_);
  handles_setup_ = true;
}

void CuDNNSoftmaxLayer::Reshape(const vector<Blob*>& bottom,
                                const vector<Blob*>& top) {
  SoftmaxLayer::Reshape(bottom, top);
  int N = this->outer_num_;
  int K = bottom[0]->shape(this->softmax_axis_);
  int H = this->inner_num_;
  int W = 1;
  cudnn::setTensor4dDesc(&bottom_desc_, N, K, H, W);
  cudnn::setTensor4dDesc(&top_desc_, N, K, H, W);
}

CuDNNSoftmaxLayer::~CuDNNSoftmaxLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);
  cudnnDestroy(handle_);
}

}  // namespace caffe

#endif  // USE_CUDNN
