#ifndef CAFFE_ROI_POOLING_LAYER_HPP_
#define CAFFE_ROI_POOLING_LAYER_HPP_

#include <vector>

#include "../layer.hpp"

namespace caffe {

/** 
 * @brief Perform max pooling on regions of interest specified by input, takes
 *        as input N feature maps and a list of R regions of interest.
 *
 *   ROIPoolingLayer takes 2 inputs and produces 1 output. bottom[0] is
 *   [N x C x H x W] feature maps on which pooling is performed. bottom[1] is
 *   [R x 5] containing a list R ROI tuples with batch index and coordinates of
 *   regions of interest. Each row in bottom[1] is a ROI tuple in format
 *   [batch_index x1 y1 x2 y2], where batch_index corresponds to the index of
 *   instance in the first input and x1 y1 x2 y2 are 0-indexed coordinates
 *   of ROI rectangle (including its boundaries).
 *
 *   For each of the R ROIs, max-pooling is performed over pooled_h x pooled_w
 *   output bins (specified in roi_pooling_param). The pooling bin sizes are
 *   adaptively set such that they tile ROI rectangle in the indexed feature
 *   map. The pooling region of vertical bin ph in [0, pooled_h) is computed as
 *
 *    start_ph (included) = y1 + floor(ph * (y2 - y1 + 1) / pooled_h)
 *    end_ph (excluded)   = y1 + ceil((ph + 1) * (y2 - y1 + 1) / pooled_h)
 *
 *   and similar horizontal bins.
 *
 * @param param provides ROIPoolingParameter roi_pooling_param,
 *        with ROIPoolingLayer options:
 *  - pooled_h. The pooled output height.
 *  - pooled_w. The pooled output width
 *  - spatial_scale. Multiplicative spatial scale factor to translate ROI
 *  coordinates from their input scale to the scale used when pooling.
 *
 * Fast R-CNN
 * Written by Ross Girshick
 */

class ROIPoolingLayer : public Layer {
 public:
  explicit ROIPoolingLayer(const LayerParameter& param)
      : Layer(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
                          const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
                       const vector<Blob*>& top);

  virtual const char* type() const { return "ROIPooling"; }

  virtual int MinBottomBlobs() const { return 2; }
  virtual int MaxBottomBlobs() const { return 2; }
  virtual int MinTopBlobs() const { return 1; }
  virtual int MaxTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);

  int channels_;
  int height_;
  int width_;
  int pooled_height_;
  int pooled_width_;
  real_t spatial_scale_;
};

}  // namespace caffe

#endif  // CAFFE_ROI_POOLING_LAYER_HPP_
