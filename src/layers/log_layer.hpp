#ifndef CAFFE_LOG_LAYER_HPP_
#define CAFFE_LOG_LAYER_HPP_

#include <vector>

#include "./neuron_layer.hpp"

namespace caffe {

/**
 * @brief Computes @f$ y = log_{\gamma}(\alpha x + \beta) @f$,
 *        as specified by the scale @f$ \alpha @f$, shift @f$ \beta @f$,
 *        and base @f$ \gamma @f$.
 */
class LogLayer : public NeuronLayer {
 public:
  /**
   * @param param provides LogParameter log_param,
   *     with LogLayer options:
   *   - scale (\b optional, default 1) the scale @f$ \alpha @f$
   *   - shift (\b optional, default 0) the shift @f$ \beta @f$
   *   - base (\b optional, default -1 for a value of @f$ e \approx 2.718 @f$)
   *         the base @f$ \gamma @f$
   */
  explicit LogLayer(const LayerParameter& param)
      : NeuronLayer(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
                          const vector<Blob*>& top);

  virtual const char* type() const { return "Log"; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the computed outputs @f$
   *        y = log_{\gamma}(\alpha x + \beta)
   *      @f$
   */
  virtual void Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);

  real_t base_scale_;
  real_t input_scale_, input_shift_;
  real_t backward_num_scale_;
};

}  // namespace caffe

#endif  // CAFFE_LOG_LAYER_HPP_
