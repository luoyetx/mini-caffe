#ifndef CAFFE_NNPACK_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_NNPACK_INNER_PRODUCT_LAYER_HPP_

#include "./nnpack.hpp"
#include "../inner_product_layer.hpp"

namespace caffe {

#ifdef USE_NNPACK
class NNPackInnerProductLayer : public InnerProductLayer {
 public:
  explicit NNPackInnerProductLayer(const LayerParameter& param)
      : InnerProductLayer(param) {}
  virtual void Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);
};
#endif  // USE_NNPACK

}  // namespace caffe

#endif  // CAFFE_NNPACK_INNER_PRODUCT_LAYER_HPP_
