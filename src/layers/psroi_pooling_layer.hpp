// ------------------------------------------------------------------
// R-FCN
// Copyright (c) 2016 Microsoft
// Licensed under The MIT License [see r-fcn/LICENSE for details]
// Written by Yi Li
// ------------------------------------------------------------------

#ifndef CAFFE_PSROI_LAYERS_HPP_
#define CAFFE_PSROI_LAYERS_HPP_

#include "../layer.hpp"

namespace caffe {

/**
 * PSROIPoolingLayer:
 *   Position-Sensitive Region of Interest Pooling Layer
 */
class PSROIPoolingLayer : public Layer {
 public:
  explicit PSROIPoolingLayer(const LayerParameter& param)
      : Layer(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
                          const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
                       const vector<Blob*>& top);

  virtual const char* type() const { return "PSROIPooling"; }

  virtual int MinBottomBlobs() const { return 2; }
  virtual int MaxBottomBlobs() const { return 2; }
  virtual int MinTopBlobs() const { return 1; }
  virtual int MaxTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);

  real_t spatial_scale_;
  int output_dim_;
  int group_size_;

  int channels_;
  int height_;
  int width_;

  int pooled_height_;
  int pooled_width_;
};

}  // namespace caffe

#endif  // CAFFE_PSROI_LAYERS_HPP_
