#ifndef CAFFE_MVN_LAYER_HPP_
#define CAFFE_MVN_LAYER_HPP_

#include <vector>

#include "../layer.hpp"

namespace caffe {

/**
 * @brief Normalizes the input to have 0-mean and/or unit (1) variance.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
class MVNLayer : public Layer {
 public:
  explicit MVNLayer(const LayerParameter& param)
      : Layer(param) {}
  virtual void Reshape(const vector<Blob*>& bottom,
                       const vector<Blob*>& top);
  virtual vector<Blob*> GetTempBlobs() { return {&mean_, &variance_, &temp_}; }

  virtual const char* type() const { return "MVN"; }
  virtual int ExactNumBottomBlobs() const { return 1; }
  virtual int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);

  Blob mean_, variance_, temp_;

  /// sum_multiplier is used to carry out sum using BLAS
  Blob sum_multiplier_;
  real_t eps_;
};

}  // namespace caffe

#endif  // CAFFE_MVN_LAYER_HPP_
