#ifndef CAFFE_CUDNN_LCN_LAYER_HPP_
#define CAFFE_CUDNN_LCN_LAYER_HPP_

#include "./cudnn.hpp"
#include "../lrn_layer.hpp"
#include "../power_layer.hpp"

namespace caffe {

#ifdef USE_CUDNN
class CuDNNLCNLayer : public LRNLayer {
 public:
  explicit CuDNNLCNLayer(const LayerParameter& param)
      : LRNLayer(param), handles_setup_(false), tempDataSize(0),
        tempData1(NULL), tempData2(NULL) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
                          const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
                       const vector<Blob*>& top);
  virtual ~CuDNNLCNLayer();

 protected:
  virtual void Forward_gpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);

  bool handles_setup_;
  cudnnHandle_t handle_;
  cudnnLRNDescriptor_t norm_desc_;
  cudnnTensorDescriptor_t bottom_desc_, top_desc_;

  int size_, pre_pad_;
  real_t alpha_, beta_, k_;

  size_t tempDataSize;
  void *tempData1, *tempData2;
};
#endif  // USE_CUDNN

}  // namespace caffe

#endif  // CAFFE_CUDNN_LCN_LAYER_HPP_
