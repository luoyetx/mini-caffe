#ifndef CAFFE_PERMUTE_LAYER_HPP_
#define CAFFE_PERMUTE_LAYER_HPP_

#include <vector>

#include "../layer.hpp"

namespace caffe {

class PermuteLayer : public Layer {
 public:
  explicit PermuteLayer(const LayerParameter& param)
      : Layer(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
                          const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
                       const vector<Blob*>& top);

  virtual inline const char* type() const { return "Permute"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);

  int num_axes_;
  bool need_permute_;

  // Use Blob because it is convenient to be accessible in .cu file.
  BlobInt permute_order_;
  BlobInt old_steps_;
  BlobInt new_steps_;
};

}  // namespace caffe

#endif  // CAFFE_PERMUTE_LAYER_HPP_
