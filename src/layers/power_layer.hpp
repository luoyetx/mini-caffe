#ifndef CAFFE_POWER_LAYER_HPP_
#define CAFFE_POWER_LAYER_HPP_

#include <vector>

#include "./neuron_layer.hpp"

namespace caffe {

/**
 * @brief Computes @f$ y = (\alpha x + \beta) ^ \gamma @f$,
 *        as specified by the scale @f$ \alpha @f$, shift @f$ \beta @f$,
 *        and power @f$ \gamma @f$.
 */
class PowerLayer : public NeuronLayer {
 public:
  /**
   * @param param provides PowerParameter power_param,
   *     with PowerLayer options:
   *   - scale (\b optional, default 1) the scale @f$ \alpha @f$
   *   - shift (\b optional, default 0) the shift @f$ \beta @f$
   *   - power (\b optional, default 1) the power @f$ \gamma @f$
   */
  explicit PowerLayer(const LayerParameter& param)
      : NeuronLayer(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
                          const vector<Blob*>& top);

  virtual const char* type() const { return "Power"; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the computed outputs @f$
   *        y = (\alpha x + \beta) ^ \gamma
   *      @f$
   */
  virtual void Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);

  /// @brief @f$ \gamma @f$ from layer_param_.power_param()
  real_t power_;
  /// @brief @f$ \alpha @f$ from layer_param_.power_param()
  real_t scale_;
  /// @brief @f$ \beta @f$ from layer_param_.power_param()
  real_t shift_;
  /// @brief Result of @f$ \alpha \gamma @f$
  real_t diff_scale_;
};

}  // namespace caffe

#endif  // CAFFE_POWER_LAYER_HPP_
