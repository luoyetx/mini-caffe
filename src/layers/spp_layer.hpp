#ifndef CAFFE_SPP_LAYER_HPP_
#define CAFFE_SPP_LAYER_HPP_

#include <vector>

#include "../layer.hpp"

namespace caffe {

/**
 * @brief Does spatial pyramid pooling on the input image
 *        by taking the max, average, etc. within regions
 *        so that the result vector of different sized
 *        images are of the same size.
 */
class SPPLayer : public Layer {
 public:
  explicit SPPLayer(const LayerParameter& param)
      : Layer(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
                          const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
                       const vector<Blob*>& top);

  virtual const char* type() const { return "SPP"; }
  virtual int ExactNumBottomBlobs() const { return 1; }
  virtual int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);
  // calculates the kernel and stride dimensions for the pooling layer,
  // returns a correctly configured LayerParameter for a PoolingLayer
  virtual LayerParameter GetPoolingParam(const int pyramid_level,
      const int bottom_h, const int bottom_w, const SPPParameter spp_param);

  int pyramid_height_;
  int bottom_h_, bottom_w_;
  int num_;
  int channels_;
  int kernel_h_, kernel_w_;
  int pad_h_, pad_w_;
  bool reshaped_first_time_;

  /// the internal Split layer that feeds the pooling layers
  shared_ptr<SplitLayer> split_layer_;
  /// top vector holder used in call to the underlying SplitLayer::Forward
  vector<Blob*> split_top_vec_;
  /// bottom vector holder used in call to the underlying PoolingLayer::Forward
  vector<vector<Blob*>*> pooling_bottom_vecs_;
  /// the internal Pooling layers of different kernel sizes
  vector<shared_ptr<PoolingLayer> > pooling_layers_;
  /// top vector holders used in call to the underlying PoolingLayer::Forward
  vector<vector<Blob*>*> pooling_top_vecs_;
  /// pooling_outputs stores the outputs of the PoolingLayers
  vector<Blob*> pooling_outputs_;
  /// the internal Flatten layers that the Pooling layers feed into
  vector<FlattenLayer*> flatten_layers_;
  /// top vector holders used in call to the underlying FlattenLayer::Forward
  vector<vector<Blob*>*> flatten_top_vecs_;
  /// flatten_outputs stores the outputs of the FlattenLayers
  vector<Blob*> flatten_outputs_;
  /// bottom vector holder used in call to the underlying ConcatLayer::Forward
  vector<Blob*> concat_bottom_vec_;
  /// the internal Concat layers that the Flatten layers feed into
  shared_ptr<ConcatLayer> concat_layer_;
};

}  // namespace caffe

#endif  // CAFFE_SPP_LAYER_HPP_
