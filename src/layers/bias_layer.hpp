#ifndef CAFFE_BIAS_LAYER_HPP_
#define CAFFE_BIAS_LAYER_HPP_

#include <vector>

#include "../layer.hpp"

namespace caffe {

/**
 * @brief Computes a sum of two input Blobs, with the shape of the
 *        latter Blob "broadcast" to match the shape of the former.
 *        Equivalent to tiling the latter Blob, then computing the elementwise
 *        sum.
 *
 * The second input may be omitted, in which case it's learned as a parameter
 * of the layer.
 */
class BiasLayer : public Layer {
 public:
  explicit BiasLayer(const LayerParameter& param)
      : Layer(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
                          const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
                       const vector<Blob*>& top);

  virtual const char* type() const { return "Bias"; }
  virtual int MinBottomBlobs() const { return 1; }
  virtual int MaxBottomBlobs() const { return 2; }
  virtual int ExactNumTopBlobs() const { return 1; }

  virtual void Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);

 private:
  Blob bias_multiplier_;
  int outer_dim_, bias_dim_, inner_dim_, dim_;
};

}  // namespace caffe

#endif  // CAFFE_BIAS_LAYER_HPP_
