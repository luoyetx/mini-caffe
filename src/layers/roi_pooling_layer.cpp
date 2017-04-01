#include <algorithm>
#include <cfloat>
#include <vector>

#include "./roi_pooling_layer.hpp"
#include "../util/math_functions.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

void ROIPoolingLayer::LayerSetUp(const vector<Blob*>& bottom,
                                 const vector<Blob*>& top) {
  ROIPoolingParameter roi_pool_param = this->layer_param_.roi_pooling_param();
  CHECK_GT(roi_pool_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(roi_pool_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = roi_pool_param.pooled_h();
  pooled_width_ = roi_pool_param.pooled_w();
  spatial_scale_ = roi_pool_param.spatial_scale();
  //LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

void ROIPoolingLayer::Reshape(const vector<Blob*>& bottom,
                              const vector<Blob*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
}

void ROIPoolingLayer::Forward_cpu(const vector<Blob*>& bottom,
                                  const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->cpu_data();
  const real_t* bottom_rois = bottom[1]->cpu_data();
  // Number of ROIs
  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num();
  int top_count = top[0]->count();
  real_t* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, static_cast<real_t>(-FLT_MAX), top_data);

  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = round(bottom_rois[1] * spatial_scale_);
    int roi_start_h = round(bottom_rois[2] * spatial_scale_);
    int roi_end_w = round(bottom_rois[3] * spatial_scale_);
    int roi_end_h = round(bottom_rois[4] * spatial_scale_);
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);

    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    const real_t bin_size_h = static_cast<real_t>(roi_height) /
                              static_cast<real_t>(pooled_height_);
    const real_t bin_size_w = static_cast<real_t>(roi_width) /
                              static_cast<real_t>(pooled_width_);

    const real_t* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);

    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          int hstart = static_cast<int>(floor(static_cast<real_t>(ph)
                                              * bin_size_h));
          int wstart = static_cast<int>(floor(static_cast<real_t>(pw)
                                              * bin_size_w));
          int hend = static_cast<int>(ceil(static_cast<real_t>(ph + 1)
                                           * bin_size_h));
          int wend = static_cast<int>(ceil(static_cast<real_t>(pw + 1)
                                           * bin_size_w));

          hstart = min(max(hstart + roi_start_h, 0), height_);
          hend = min(max(hend + roi_start_h, 0), height_);
          wstart = min(max(wstart + roi_start_w, 0), width_);
          wend = min(max(wend + roi_start_w, 0), width_);

          bool is_empty = (hend <= hstart) || (wend <= wstart);

          const int pool_index = ph * pooled_width_ + pw;
          if (is_empty) {
            top_data[pool_index] = 0;
          }

          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = h * width_ + w;
              top_data[pool_index] = max(top_data[pool_index], batch_data[index]);
            }
          }
        }
      }
      // Increment all data pointers by one channel
      batch_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
    }
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }
}

#ifndef USE_CUDA
STUB_GPU(ROIPoolingLayer);
#endif

REGISTER_LAYER_CLASS(ROIPooling);

}  // namespace caffe
