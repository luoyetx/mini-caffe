#ifndef CAFFE_NNPACK_POOLING_LAYER_HPP_
#define CAFFE_NNPACK_POOLING_LAYER_HPP_

#include "./nnpack.hpp"
#include "../pooling_layer.hpp"

namespace caffe {

#ifdef USE_NNPACK
class NNPackPoolingLayer : public PoolingLayer {
 public:
  explicit NNPackPoolingLayer(const LayerParameter& param)
      : PoolingLayer(param) {}
  virtual void Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);
};
#endif  // USE_NNPACK

}  // namespace caffe

#endif  // CAFFE_NNPACK_POOLING_LAYER_HPP_
