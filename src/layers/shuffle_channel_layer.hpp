#ifndef CAFFE_SHUFFLE_CHANNEL_LAYER_HPP_
#define CAFFE_SHUFFLE_CHANNEL_LAYER_HPP_

#include <vector>

#include "../layer.hpp"

namespace caffe {

class ShuffleChannelLayer : public Layer {
 public:
  explicit ShuffleChannelLayer(const LayerParameter& param)
      : Layer(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
                          const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
                       const vector<Blob*>& top);
  virtual inline const char* type() const { return "ShuffleChannel"; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
                            const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
                            const vector<Blob*>& top);

  //Blob<Dtype> temp_blob_;
  int group_;
};

}  // namespace caffe

#endif  // CAFFE_SHUFFLE_CHANNEL_LAYER_HPP_
