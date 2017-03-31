// ------------------------------------------------------------------
// R-FCN
// Copyright (c) 2016 Microsoft
// Licensed under The MIT License [see r-fcn/LICENSE for details]
// Written by Yi Li
// ------------------------------------------------------------------

#include "./psroi_pooling_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

__global__ void PSROIPoolingForward(const int nthreads,
                                    const real_t* bottom_data,
                                    const real_t spatial_scale,
                                    const int channels,
                                    const int height, const int width,
                                    const int pooled_height, const int pooled_width,
                                    const real_t* bottom_rois,
                                    const int output_dim,
                                    const int group_size,
                                    real_t* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;

    // [start, end) interval for spatial sampling
    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    real_t roi_start_w = static_cast<real_t>(round(bottom_rois[1])) * spatial_scale;
    real_t roi_start_h = static_cast<real_t>(round(bottom_rois[2])) * spatial_scale;
    real_t roi_end_w = static_cast<real_t>(round(bottom_rois[3]) + 1.) * spatial_scale;
    real_t roi_end_h = static_cast<real_t>(round(bottom_rois[4]) + 1.) * spatial_scale;

    // Force too small ROIs to be 1x1
    real_t roi_width = max(roi_end_w - roi_start_w, static_cast<real_t>(0.1)); //avoid 0
    real_t roi_height = max(roi_end_h - roi_start_h, static_cast<real_t>(0.1));

    // Compute w and h at bottom
    real_t bin_size_h = roi_height / static_cast<real_t>(pooled_height);
    real_t bin_size_w = roi_width / static_cast<real_t>(pooled_width);

    int hstart = floor(static_cast<real_t>(ph) * bin_size_h + roi_start_h);
    int wstart = floor(static_cast<real_t>(pw)* bin_size_w + roi_start_w);
    int hend = ceil(static_cast<real_t>(ph + 1) * bin_size_h + roi_start_h);
    int wend = ceil(static_cast<real_t>(pw + 1) * bin_size_w + roi_start_w);
    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart, 0), height);
    hend = min(max(hend, 0), height);
    wstart = min(max(wstart, 0),width);
    wend = min(max(wend, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    int gw = pw;
    int gh = ph;
    int c = (ctop*group_size + gh)*group_size + gw;

    bottom_data += (roi_batch_ind * channels + c) * height * width;
    real_t out_sum = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h*width + w;
        out_sum += bottom_data[bottom_index];
      }
    }

    real_t bin_area = (hend - hstart)*(wend - wstart);
    top_data[index] = is_empty ? 0. : out_sum/bin_area;
  }
}

void PSROIPoolingLayer::Forward_gpu(const vector<Blob*>& bottom,
                                    const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->gpu_data();
  const real_t* bottom_rois = bottom[1]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  caffe_gpu_set(count, static_cast<real_t>(0), top_data);
  // NOLINT_NEXT_LINE(whitespace/operators)
  PSROIPoolingForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, spatial_scale_, channels_, height_, width_, pooled_height_,
      pooled_width_, bottom_rois, output_dim_, group_size_, top_data);
  CUDA_POST_KERNEL_CHECK;
}

}  // namespace caffe
