#ifndef CAFFE_BATCHNORM_LAYER_HPP_
#define CAFFE_BATCHNORM_LAYER_HPP_

#include <vector>

#include "../layer.hpp"

namespace caffe {

/**
 * @brief Normalizes the input to have 0-mean and/or unit (1) variance across
 *        the batch.
 *
 * This layer computes Batch Normalization described in [1].  For
 * each channel in the data (i.e. axis 1), it subtracts the mean and divides
 * by the variance, where both statistics are computed across both spatial
 * dimensions and across the different examples in the batch.
 *
 * By default, during training time, the network is computing global mean/
 * variance statistics via a running average, which is then used at test
 * time to allow deterministic outputs for each input.  You can manually
 * toggle whether the network is accumulating or using the statistics via the
 * use_global_stats option.  IMPORTANT: for this feature to work, you MUST
 * set the learning rate to zero for all three parameter blobs, i.e.,
 * param {lr_mult: 0} three times in the layer definition.
 *
 * Note that the original paper also included a per-channel learned bias and
 * scaling factor.  It is possible (though a bit cumbersome) to implement
 * this in caffe using a single-channel DummyDataLayer filled with zeros,
 * followed by a Convolution layer with output the same size as the current.
 * This produces a channel-specific value that can be added or multiplied by
 * the BatchNorm layer's output.
 *
 * [1] S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network
 *     Training by Reducing Internal Covariate Shift." arXiv preprint
 *     arXiv:1502.03167 (2015).
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
class BatchNormLayer : public Layer {
 public:
  explicit BatchNormLayer(const LayerParameter& param)
      : Layer(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
                          const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
                       const vector<Blob*>& top);
  virtual vector<Blob*> GetTempBlobs() { return {&temp_}; }

  virtual const char* type() const { return "BatchNorm"; }
  virtual int ExactNumBottomBlobs() const { return 1; }
  virtual int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);

  Blob mean_, variance_, temp_;
  bool use_global_stats_;
  int channels_;
  real_t eps_;

  // extra temporarary variables is used to carry out sums/broadcasting
  // using BLAS
  Blob batch_sum_multiplier_;
  Blob num_by_chans_;
  Blob spatial_sum_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_BATCHNORM_LAYER_HPP_
