#ifndef CAFFE_PRELU_LAYER_HPP_
#define CAFFE_PRELU_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "./neuron_layer.hpp"

namespace caffe {

/**
 * @brief Parameterized Rectified Linear Unit non-linearity @f$
 *        y_i = \max(0, x_i) + a_i \min(0, x_i)
 *        @f$. The differences from ReLULayer are 1) negative slopes are
 *        learnable though backprop and 2) negative slopes can vary across
 *        channels. The number of axes of input blob should be greater than or
 *        equal to 2. The 1st axis (0-based) is seen as channels.
 */
template <typename Dtype>
class PReLULayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param param provides PReLUParameter prelu_param,
   *     with PReLULayer options:
   *   - filler (\b optional, FillerParameter,
   *     default {'type': constant 'value':0.25}).
   *   - channel_shared (\b optional, default false).
   *     negative slopes are shared across channels.
   */
  explicit PReLULayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PReLU"; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times ...) @f$
   *      the inputs @f$ x @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times C \times ...) @f$
   *      the computed outputs for each channel @f$i@f$ @f$
   *        y_i = \max(0, x_i) + a_i \min(0, x_i)
   *      @f$.
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  bool channel_shared_;
  Blob<Dtype> multiplier_;  // dot multiplier for backward computation of params
  Blob<Dtype> backward_buff_;  // temporary buffer for backward computation
  Blob<Dtype> bottom_memory_;  // memory for in-place computation
};

}  // namespace caffe

#endif  // CAFFE_PRELU_LAYER_HPP_
