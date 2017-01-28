#ifndef CAFFE_CUDNN_LRN_LAYER_HPP_
#define CAFFE_CUDNN_LRN_LAYER_HPP_

#include "./cudnn.hpp"
#include "../lrn_layer.hpp"

namespace caffe {

#ifdef USE_CUDNN
class CuDNNLRNLayer : public LRNLayer {
 public:
  explicit CuDNNLRNLayer(const LayerParameter& param)
      : LRNLayer(param), handles_setup_(false) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
                          const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
                       const vector<Blob*>& top);
  virtual ~CuDNNLRNLayer();

 protected:
  virtual void Forward_gpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);

  bool handles_setup_;
  cudnnHandle_t handle_;
  cudnnLRNDescriptor_t norm_desc_;
  cudnnTensorDescriptor_t bottom_desc_, top_desc_;

  int size_;
  real_t alpha_, beta_, k_;
};
#endif  // USE_CUDNN

}  // namespace caffe

#endif  // CAFFE_CUDNN_LRN_LAYER_HPP_
