#ifndef CAFFE_INFOGAIN_LOSS_LAYER_HPP_
#define CAFFE_INFOGAIN_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief A generalization of MultinomialLogisticLossLayer that takes an
 *        "information gain" (infogain) matrix specifying the "value" of all label
 *        pairs.
 *
 * Equivalent to the MultinomialLogisticLossLayer if the infogain matrix is the
 * identity.
 *
 * @param bottom input Blob vector (length 2-3)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the predictions @f$ \hat{p} @f$, a Blob with values in
 *      @f$ [0, 1] @f$ indicating the predicted probability of each of the
 *      @f$ K = CHW @f$ classes.  Each prediction vector @f$ \hat{p}_n @f$
 *      should sum to 1 as in a probability distribution: @f$
 *      \forall n \sum\limits_{k=1}^K \hat{p}_{nk} = 1 @f$.
 *   -# @f$ (N \times 1 \times 1 \times 1) @f$
 *      the labels @f$ l @f$, an integer-valued Blob with values
 *      @f$ l_n \in [0, 1, 2, ..., K - 1] @f$
 *      indicating the correct class label among the @f$ K @f$ classes
 *   -# @f$ (1 \times 1 \times K \times K) @f$
 *      (\b optional) the infogain matrix @f$ H @f$.  This must be provided as
 *      the third bottom blob input if not provided as the infogain_mat in the
 *      InfogainLossParameter. If @f$ H = I @f$, this layer is equivalent to the
 *      MultinomialLogisticLossLayer.
 * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the computed infogain multinomial logistic loss: @f$ E =
 *        \frac{-1}{N} \sum\limits_{n=1}^N H_{l_n} \log(\hat{p}_n) =
 *        \frac{-1}{N} \sum\limits_{n=1}^N \sum\limits_{k=1}^{K} H_{l_n,k}
 *        \log(\hat{p}_{n,k})
 *      @f$, where @f$ H_{l_n} @f$ denotes row @f$l_n@f$ of @f$H@f$.
 */
template <typename Dtype>
class InfogainLossLayer : public LossLayer<Dtype> {
 public:
  explicit InfogainLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), infogain_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // InfogainLossLayer takes 2-3 bottom Blobs; if there are 3 the third should
  // be the infogain matrix.  (Otherwise the infogain matrix is loaded from a
  // file specified by LayerParameter.)
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }

  virtual inline const char* type() const { return "InfogainLoss"; }

 protected:
  /// @copydoc InfogainLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  Blob<Dtype> infogain_;
};

}  // namespace caffe

#endif  // CAFFE_INFOGAIN_LOSS_LAYER_HPP_
