// ------------------------------------------------------------------
// R-FCN
// Copyright (c) 2016 Microsoft
// Licensed under The MIT License [see r-fcn/LICENSE for details]
// Written by Yi Li
// ------------------------------------------------------------------

#include "./psroi_pooling_layer.hpp"
#include "../util/math_functions.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

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
  const real_t* bottom_data = bottom[0]->cpu_data();
  const real_t* bottom_rois = bottom[1]->cpu_data();
  // Number of ROIs
  const int num_rois = bottom[1]->num();
  const int batch_size = bottom[0]->num();
  const int top_count = top[0]->count();
  real_t* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, static_cast<real_t>(0), top_data);

  // For each ROI R = [batch_index x1, y1, x2, y2]: avg pool over R
  for (int n = 0; n < num_rois; ++n) {
    const int roi_batch_ind = bottom_rois[0];
    const real_t roi_start_w = static_cast<real_t>(round(bottom_rois[1]) * spatial_scale_);
    const real_t roi_start_h = static_cast<real_t>(round(bottom_rois[2]) * spatial_scale_);
    const real_t roi_end_w = static_cast<real_t>(round(bottom_rois[3] + 1) * spatial_scale_);
    const real_t roi_end_h = static_cast<real_t>(round(bottom_rois[4] + 1) * spatial_scale_);
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);

    const real_t roi_height = max(roi_end_h - roi_start_h, static_cast<real_t>(0.1));
    const real_t roi_width = max(roi_end_w - roi_start_w, static_cast<real_t>(0.1));
    const real_t bin_size_h = roi_height / static_cast<real_t>(pooled_height_);
    const real_t bin_size_w = roi_width / static_cast<real_t>(pooled_width_);

    const real_t* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);
    for (int c = 0; c < output_dim_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          int hstart = static_cast<int>(floor(static_cast<real_t>(ph)*bin_size_h+roi_start_h));
          int wstart = static_cast<int>(floor(static_cast<real_t>(pw)*bin_size_w+roi_start_w));
          int hend = static_cast<int>(ceil(static_cast<real_t>(ph+1)*bin_size_h+roi_start_h));
          int wend = static_cast<int>(ceil(static_cast<real_t>(pw+1)*bin_size_w+roi_start_w));
          hstart = min(max(hstart, 0), height_);
          hend = min(max(hend, 0), height_);
          wstart = min(max(wstart, 0), width_);
          wend = min(max(wend, 0), width_);

          const bool is_empty = (hend <= hstart) || (wend <= wstart);
          const int gw = pw;
          const int gh = ph;
          const int fm_c = (c*group_size_ + gh)*group_size_ + gw;
          const real_t* fm_data = batch_data + fm_c * height_ * width_;
          real_t out_sum = 0;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int bottom_index = h*width_ + w;
              out_sum += fm_data[bottom_index];
            }
          }
          const int bin_area = (hend - hstart)*(wend - wstart);
          const int pool_idx = ph * pooled_width_ + pw;
          out_sum = is_empty ? 0. : out_sum / static_cast<real_t>(bin_area);
          top_data[pool_idx] = out_sum;
        }
      }
      top_data += top[0]->offset(0, 1);
    }
    bottom_rois += bottom[1]->offset(1);
  }
}

#ifndef USE_CUDA
STUB_GPU(PSROIPoolingLayer);
#endif

REGISTER_LAYER_CLASS(PSROIPooling);

}  // namespace caffe
