#ifndef CAFFE_CUDNN_LRN_LAYER_HPP_
#define CAFFE_CUDNN_LRN_LAYER_HPP_

#include "./lrn_layer.hpp"
#include "../util/cudnn.hpp"
#include "../proto/caffe.pb.h"

namespace caffe {

#ifdef USE_CUDNN
template <typename Dtype>
class CuDNNLRNLayer : public LRNLayer<Dtype> {
 public:
  explicit CuDNNLRNLayer(const LayerParameter& param)
      : LRNLayer<Dtype>(param), handles_setup_(false) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~CuDNNLRNLayer();

 protected:
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  bool handles_setup_;
  cudnnHandle_t handle_;
  cudnnLRNDescriptor_t norm_desc_;
  cudnnTensorDescriptor_t bottom_desc_, top_desc_;

  int size_;
  Dtype alpha_, beta_, k_;
};
#endif

}  // namespace caffe

#endif  // CAFFE_CUDNN_LRN_LAYER_HPP_
