#ifndef CAFFE_PARAMETER_LAYER_HPP_
#define CAFFE_PARAMETER_LAYER_HPP_

#include <vector>

#include "../layer.hpp"

namespace caffe {

class ParameterLayer : public Layer {
 public:
  explicit ParameterLayer(const LayerParameter& param)
      : Layer(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
                          const vector<Blob*>& top) {
    if (this->blobs_.size() > 0) {
      LOG(INFO) << "Skipping parameter initialization";
    } else {
      this->blobs_.resize(1);
      this->blobs_[0].reset(new Blob());
      this->blobs_[0]->Reshape(this->layer_param_.parameter_param().shape());
    }
    top[0]->Reshape(this->layer_param_.parameter_param().shape());
  }
  virtual void Reshape(const vector<Blob*>& bottom,
                       const vector<Blob*>& top) {}
  virtual const char* type() const { return "Parameter"; }
  virtual int ExactNumBottomBlobs() const { return 0; }
  virtual int ExactNumTopBlobs() const { return 1; }

 protected:
   virtual void Forward_cpu(const vector<Blob*>& bottom,
                            const vector<Blob*>& top);
   virtual void Forward_gpu(const vector<Blob*>& bottom,
                            const vector<Blob*>& top);
};

}  // namespace caffe

#endif
