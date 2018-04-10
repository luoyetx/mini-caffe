#ifndef CAFFE_CONV_DW_LAYER_HPP_
#define CAFFE_CONV_DW_LAYER_HPP_

#include <vector>

#include "../layer.hpp"

namespace caffe {

class ConvolutionDepthwiseLayer : public Layer {
 public:
  explicit ConvolutionDepthwiseLayer(const LayerParameter& param)
      : Layer(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
                          const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
                       const vector<Blob*>& top);
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline const char* type() const { return "ConvolutionDepthwise"; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);
  unsigned int kernel_h_;
  unsigned int kernel_w_;
  unsigned int stride_h_;
  unsigned int stride_w_;
  unsigned int pad_h_;
  unsigned int pad_w_;
  unsigned int dilation_h_;
  unsigned int dilation_w_;
};

}  // namespace caffe

#endif  // CAFFE_CONV_DW_LAYER_HPP_
