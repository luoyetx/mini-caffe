// ------------------------------------------------------------------
// R-FCN
// Copyright (c) 2016 Microsoft
// Licensed under The MIT License [see r-fcn/LICENSE for details]
// Written by Yi Li
// ------------------------------------------------------------------

#include "./psroi_pooling_layer.hpp"

namespace caffe {

void PSROIPoolingLayer::LayerSetUp(const vector<Blob*>& bottom,
                                   const vector<Blob*>& top){
  PSROIPoolingParameter psroi_pooling_param = this->layer_param_.psroi_pooling_param();
  spatial_scale_ = psroi_pooling_param.spatial_scale();
  //LOG(INFO) << "Spatial scale: " << spatial_scale_;

  CHECK_GT(psroi_pooling_param.output_dim(), 0)
      << "output_dim must be > 0";
  CHECK_GT(psroi_pooling_param.group_size(), 0)
      << "group_size must be > 0";

  output_dim_ = psroi_pooling_param.output_dim();
  group_size_ = psroi_pooling_param.group_size();
  pooled_height_ = group_size_;
  pooled_width_ = group_size_;
}

void PSROIPoolingLayer::Reshape(const vector<Blob*>& bottom,
                                const vector<Blob*>& top) {
  channels_ = bottom[0]->channels();
  CHECK_EQ(channels_, output_dim_*group_size_*group_size_)
      << "input channel number does not match layer parameters";
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), output_dim_, pooled_height_, pooled_width_);
}

void PSROIPoolingLayer::Forward_cpu(const vector<Blob*>& bottom,
                                    const vector<Blob*>& top) {
  NOT_IMPLEMENTED;
}

#ifndef USE_CUDA
STUB_GPU(PSROIPoolingLayer);
#endif

REGISTER_LAYER_CLASS(PSROIPooling);

}  // namespace caffe
