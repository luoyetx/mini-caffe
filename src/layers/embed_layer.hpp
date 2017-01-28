#ifndef CAFFE_EMBED_LAYER_HPP_
#define CAFFE_EMBED_LAYER_HPP_

#include <vector>

#include "../layer.hpp"

namespace caffe {

/**
 * @brief A layer for learning "embeddings" of one-hot vector input.
 *        Equivalent to an InnerProductLayer with one-hot vectors as input, but
 *        for efficiency the input is the "hot" index of each column itself.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
class EmbedLayer : public Layer {
 public:
  explicit EmbedLayer(const LayerParameter& param)
      : Layer(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
                          const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
                       const vector<Blob*>& top);

  virtual const char* type() const { return "Embed"; }
  virtual int ExactNumBottomBlobs() const { return 1; }
  virtual int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob bias_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_EMBED_LAYER_HPP_
