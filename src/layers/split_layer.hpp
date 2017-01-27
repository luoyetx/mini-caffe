#ifndef CAFFE_SPLIT_LAYER_HPP_
#define CAFFE_SPLIT_LAYER_HPP_

#include <vector>

#include "../layer.hpp"

namespace caffe {

/**
 * @brief Creates a "split" path in the network by copying the bottom Blob
 *        into multiple top Blob%s to be used by multiple consuming layers.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
class SplitLayer : public Layer {
 public:
  explicit SplitLayer(const LayerParameter& param)
      : Layer(param) {}
  virtual void Reshape(const vector<Blob*>& bottom,
                       const vector<Blob*>& top);

  virtual const char* type() const { return "Split"; }
  virtual int ExactNumBottomBlobs() const { return 1; }
  virtual int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);

  int count_;
};

}  // namespace caffe

#endif  // CAFFE_SPLIT_LAYER_HPP_
