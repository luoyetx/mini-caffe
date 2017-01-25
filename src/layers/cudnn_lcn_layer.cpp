#ifdef USE_CUDNN

#include <vector>

#include "./cudnn_lcn_layer.hpp"

namespace caffe {

void CuDNNLCNLayer::LayerSetUp(const vector<Blob*>& bottom,
                               const vector<Blob*>& top) {
  LRNLayer::LayerSetUp(bottom, top);

  CUDNN_CHECK(cudnnCreate(&handle_));
  CUDNN_CHECK(cudnnCreateLRNDescriptor(&norm_desc_));
  cudnn::createTensor4dDesc(&bottom_desc_);
  cudnn::createTensor4dDesc(&top_desc_);

  // create a LRN handle
  handles_setup_ = true;

  size_ = this->layer_param().lrn_param().local_size();
  pre_pad_ = (size_ - 1) / 2;
  alpha_ = this->layer_param().lrn_param().alpha();
  beta_ = this->layer_param().lrn_param().beta();
  k_ = this->layer_param().lrn_param().k();
}

void CuDNNLCNLayer::Reshape(const vector<Blob*>& bottom,
                            const vector<Blob*>& top) {
  LRNLayer::Reshape(bottom, top);
  cudnn::setTensor4dDesc(&bottom_desc_, bottom[0]->num(),
      this->channels_, this->height_, this->width_);
  cudnn::setTensor4dDesc(&top_desc_, bottom[0]->num(),
      this->channels_, this->height_, this->width_);
  CUDNN_CHECK(cudnnSetLRNDescriptor(norm_desc_, size_, alpha_, beta_, k_));

  // allocate / reallocate tempData buffers
  size_t totalSizeInBytes = sizeof(real_t)*bottom[0]->num()* \
                            this->channels_*this->height_*this->width_;

  if (totalSizeInBytes > tempDataSize) {
    tempDataSize = totalSizeInBytes;

    cudaFree(tempData1);
    cudaFree(tempData2);

    // allocate new buffers
    CUDA_CHECK(cudaMalloc(&tempData1, totalSizeInBytes));
    CUDA_CHECK(cudaMalloc(&tempData2, totalSizeInBytes));
  }
}

CuDNNLCNLayer::~CuDNNLCNLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);

  // destroy LRN handle
  cudnnDestroy(handle_);

  // free temp buffers
  cudaFree(tempData1);
  cudaFree(tempData2);
}

}  // namespace caffe

#endif  // USE_CUDNN
