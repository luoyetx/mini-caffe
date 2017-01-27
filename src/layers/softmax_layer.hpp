#ifndef CAFFE_SOFTMAX_LAYER_HPP_
#define CAFFE_SOFTMAX_LAYER_HPP_

#include <vector>

#include "../layer.hpp"

namespace caffe {

/**
 * @brief Computes the softmax function.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
class SoftmaxLayer : public Layer {
 public:
  explicit SoftmaxLayer(const LayerParameter& param)
      : Layer(param) {}
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual const char* type() const { return "Softmax"; }
  virtual int ExactNumBottomBlobs() const { return 1; }
  virtual int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);

  int outer_num_;
  int inner_num_;
  int softmax_axis_;
  /// sum_multiplier is used to carry out sum using BLAS
  Blob sum_multiplier_;
  /// scale is an intermediate Blob to hold temporary results.
  Blob scale_;
};

}  // namespace caffe

#endif  // CAFFE_SOFTMAX_LAYER_HPP_
