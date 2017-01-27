#ifndef CAFFE_ELU_LAYER_HPP_
#define CAFFE_ELU_LAYER_HPP_

#include <vector>

#include "./neuron_layer.hpp"

namespace caffe {

/**
 * @brief Exponential Linear Unit non-linearity @f$
 *        y = \left\{
 *        \begin{array}{lr}
 *            x                  & \mathrm{if} \; x > 0 \\
 *            \alpha (\exp(x)-1) & \mathrm{if} \; x \le 0
 *        \end{array} \right.
 *      @f$.
 */
class ELULayer : public NeuronLayer {
 public:
  /**
   * @param param provides ELUParameter elu_param,
   *     with ELULayer options:
   *   - alpha (\b optional, default 1).
   *     the value @f$ \alpha @f$ by which controls saturation for negative inputs.
   */
  explicit ELULayer(const LayerParameter& param)
      : NeuronLayer(param) {}

  virtual const char* type() const { return "ELU"; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the computed outputs @f$
   *        y = \left\{
   *        \begin{array}{lr}
   *            x                  & \mathrm{if} \; x > 0 \\
   *            \alpha (\exp(x)-1) & \mathrm{if} \; x \le 0
   *        \end{array} \right.
   *      @f$.
   */
  virtual void Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);
};

}  // namespace caffe

#endif  // CAFFE_ELU_LAYER_HPP_
