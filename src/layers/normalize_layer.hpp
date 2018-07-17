#ifndef CAFFE_NORMALIZE_LAYER_HPP_
#define CAFFE_NORMALIZE_LAYER_HPP_

#include <vector>

#include "../layer.hpp"

namespace caffe {

/**
 * @brief Normalizes the input to have L_p norm of 1 with scale learnable.
 *
 * TODO(weiliu89): thorough documentation for Forward, Backward, and proto params.
 */
class NormalizeLayer : public Layer {
 public:
  explicit NormalizeLayer(const LayerParameter& param)
      : Layer(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
                          const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
                       const vector<Blob*>& top);
  virtual vector<Blob*> GetTempBlobs() { return {&buffer_, &buffer_spatial_, &norm_, &sum_spatial_multiplier_}; }

  virtual inline const char* type() const { return "Normalize"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);

  Blob norm_;
  Blob sum_channel_multiplier_, sum_spatial_multiplier_;
  Blob buffer_, buffer_channel_, buffer_spatial_;
  bool across_spatial_;
  bool channel_shared_;
  real_t eps_;
};

}  // namespace caffe

#endif  // CAFFE_MVN_LAYER_HPP_
