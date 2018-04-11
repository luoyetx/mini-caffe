#include <vector>
#include "./conv_dw_layer.hpp"
//#include "caffe/util/gpu_util.cuh"

namespace caffe {

template <typename Dtype>
__global__ void ConvolutionDepthwiseWeightForward(const int nthreads,
    const Dtype* const bottom_data, const Dtype* const weight_data, const int num, const int channels,
    const int top_height, const int top_width, const int bottom_height, const int bottom_width,
    const int kernel_h, const int kernel_w, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, const int dilation_h, const int dilation_w,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / channels / top_height / top_width;
    const int c = (index / top_height / top_width) % channels;
    const int h = (index / top_width) % top_height;
    const int w = index % top_width;
    const Dtype* weight = weight_data + c * kernel_h * kernel_w;
    Dtype value = 0;
    for (int kh = 0; kh < kernel_h; ++kh)
    {
      for (int kw = 0; kw < kernel_w; ++kw)
      {
        const int h_in = -pad_h + h * stride_h + kh * dilation_h;
        const int w_in = -pad_w + w * stride_w + kw * dilation_w;
        if ((h_in >= 0) && (h_in < bottom_height) && (w_in >= 0) && (w_in < bottom_width))
        {
          const int offset = ((n * channels + c) * bottom_height + h_in) * bottom_width + w_in;
          value += (*weight) * bottom_data[offset];
        }
        ++weight;
      }
    }
    top_data[index] = value;
  }
}

template <typename Dtype>
__global__ void ConvolutionDepthwiseBiasForward(const int nthreads,
    const Dtype* const bias_data, const int num, const int channels,
    const int top_height, const int top_width, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int c = (index / top_height / top_width) % channels;
    top_data[index] += bias_data[c];
  }
}

void ConvolutionDepthwiseLayer::Forward_gpu(const vector<Blob*>& bottom,
                                            const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  const real_t* weight_data = this->blobs_[0]->gpu_data();
  const int count = top[0]->count();
  const int num = top[0]->num();
  const int channels = top[0]->channels();
  const int top_height = top[0]->height();
  const int top_width = top[0]->width();
  const int bottom_height = bottom[0]->height();
  const int bottom_width = bottom[0]->width();
  ConvolutionDepthwiseWeightForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, weight_data, num, channels,
      top_height, top_width, bottom_height, bottom_width,
      kernel_h_, kernel_w_, stride_h_, stride_w_,
      pad_h_, pad_w_, dilation_h_, dilation_w_, top_data);
  if (this->layer_param_.convolution_param().bias_term()) {
    const real_t* bias_data = this->blobs_[1]->gpu_data();
    ConvolutionDepthwiseBiasForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bias_data, num, channels,
        top_height, top_width, top_data);
  }
}

}  // namespace caffe
