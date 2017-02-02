#ifndef CAFFE_NNPACK_CONV_LAYER_HPP_
#define CAFFE_NNPACK_CONV_LAYER_HPP_

#include "./nnpack.hpp"
#include "../conv_layer.hpp"

namespace caffe {

#ifdef USE_NNPACK
class NNPackConvolutionLayer : public ConvolutionLayer {
 public:
  explicit NNPackConvolutionLayer(const LayerParameter& param)
      : ConvolutionLayer(param) {}
  virtual void Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);
};
#endif  // USE_NNPACK

}  // namespace caffe

#endif  // CAFFE_NNPACK_CONV_LAYER_HPP_