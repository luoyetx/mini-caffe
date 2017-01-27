#ifndef CAFFE_LRN_LAYER_HPP_
#define CAFFE_LRN_LAYER_HPP_

#include <vector>

#include "./eltwise_layer.hpp"
#include "./pooling_layer.hpp"
#include "./power_layer.hpp"
#include "./split_layer.hpp"

namespace caffe {

/**
 * @brief Normalize the input in a local region across or within feature maps.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
class LRNLayer : public Layer {
 public:
  explicit LRNLayer(const LayerParameter& param)
      : Layer(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
                          const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
                       const vector<Blob*>& top);

  virtual const char* type() const { return "LRN"; }
  virtual int ExactNumBottomBlobs() const { return 1; }
  virtual int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);

  virtual void CrossChannelForward_cpu(const vector<Blob*>& bottom,
                                       const vector<Blob*>& top);
  virtual void CrossChannelForward_gpu(const vector<Blob*>& bottom,
                                       const vector<Blob*>& top);
  virtual void WithinChannelForward(const vector<Blob*>& bottom,
                                    const vector<Blob*>& top);

  int size_;
  int pre_pad_;
  real_t alpha_;
  real_t beta_;
  real_t k_;
  int num_;
  int channels_;
  int height_;
  int width_;

  // Fields used for normalization ACROSS_CHANNELS
  // scale_ stores the intermediate summing results
  Blob scale_;

  // Fields used for normalization WITHIN_CHANNEL
  shared_ptr<SplitLayer> split_layer_;
  vector<Blob*> split_top_vec_;
  shared_ptr<PowerLayer> square_layer_;
  Blob square_input_;
  Blob square_output_;
  vector<Blob*> square_bottom_vec_;
  vector<Blob*> square_top_vec_;
  shared_ptr<PoolingLayer> pool_layer_;
  Blob pool_output_;
  vector<Blob*> pool_top_vec_;
  shared_ptr<PowerLayer> power_layer_;
  Blob power_output_;
  vector<Blob*> power_top_vec_;
  shared_ptr<EltwiseLayer> product_layer_;
  Blob product_input_;
  vector<Blob*> product_bottom_vec_;
};

}  // namespace caffe

#endif  // CAFFE_LRN_LAYER_HPP_
