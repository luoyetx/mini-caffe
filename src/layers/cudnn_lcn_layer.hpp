#ifndef CAFFE_CUDNN_LCN_LAYER_HPP_
#define CAFFE_CUDNN_LCN_LAYER_HPP_

#include "./lrn_layer.hpp"
#include "./power_layer.hpp"
#include "../util/cudnn.hpp"
#include "../proto/caffe.pb.h"

namespace caffe {

#ifdef USE_CUDNN
template <typename Dtype>
class CuDNNLCNLayer : public LRNLayer<Dtype> {
 public:
  explicit CuDNNLCNLayer(const LayerParameter& param)
      : LRNLayer<Dtype>(param), handles_setup_(false), tempDataSize(0),
        tempData1(NULL), tempData2(NULL) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~CuDNNLCNLayer();

 protected:
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  bool handles_setup_;
  cudnnHandle_t handle_;
  cudnnLRNDescriptor_t norm_desc_;
  cudnnTensorDescriptor_t bottom_desc_, top_desc_;

  int size_, pre_pad_;
  Dtype alpha_, beta_, k_;

  size_t tempDataSize;
  void *tempData1, *tempData2;
};
#endif

}  // namespace caffe

#endif  // CAFFE_CUDNN_LCN_LAYER_HPP_
