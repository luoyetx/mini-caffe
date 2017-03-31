#include <algorithm>
#include <cfloat>
#include <vector>

#include "./roi_pooling_layer.hpp"


using std::max;
using std::min;

namespace caffe {

__global__ void ROIPoolForward(const int nthreads, const real_t* bottom_data,
                               const real_t spatial_scale, const int channels, const int height,
                               const int width, const int pooled_height, const int pooled_width,
                               const real_t* bottom_rois, real_t* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = round(bottom_rois[1] * spatial_scale);
    int roi_start_h = round(bottom_rois[2] * spatial_scale);
    int roi_end_w = round(bottom_rois[3] * spatial_scale);
    int roi_end_h = round(bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    real_t bin_size_h = static_cast<real_t>(roi_height) /
                        static_cast<real_t>(pooled_height);
    real_t bin_size_w = static_cast<real_t>(roi_width) /
                        static_cast<real_t>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<real_t>(ph)
                                        * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<real_t>(pw)
                                        * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<real_t>(ph + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<real_t>(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    real_t maxval = is_empty ? 0 : -FLT_MAX;
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h * width + w;
        maxval = max(maxval, bottom_data[bottom_index]);
      }
    }
    top_data[index] = maxval;
  }
}

void ROIPoolingLayer::Forward_gpu(const vector<Blob*>& bottom,
                                  const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->gpu_data();
  const real_t* bottom_rois = bottom[1]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ROIPoolForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, spatial_scale_, channels_, height_, width_,
      pooled_height_, pooled_width_, bottom_rois, top_data);
  CUDA_POST_KERNEL_CHECK;
}

}  // namespace caffe
