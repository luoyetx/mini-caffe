#include <algorithm>
#include <vector>

#include "../filler.hpp"
#include "./conv_dw_layer.hpp"

namespace caffe {

void ConvolutionDepthwiseLayer::LayerSetUp(const vector<Blob*>& bottom,
                                           const vector<Blob*>& top) {
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  if (conv_param.has_kernel_h() && conv_param.has_kernel_w()) {
    kernel_h_ = conv_param.kernel_h();
    kernel_w_ = conv_param.kernel_w();
  } else {
    if (conv_param.kernel_size_size() == 1) {
      kernel_h_ = conv_param.kernel_size(0);
      kernel_w_ = conv_param.kernel_size(0);
    } else {
      kernel_h_ = conv_param.kernel_size(0);
      kernel_w_ = conv_param.kernel_size(1);
    }
  }
  if (conv_param.has_stride_h() && conv_param.has_stride_w()) {
    stride_h_ = conv_param.stride_h();
    stride_w_ = conv_param.stride_w();
  } else {
    if (conv_param.stride_size() == 0) {
      stride_h_ = 1;
      stride_w_ = 1;
    }
    else if (conv_param.stride_size() == 1) {
      stride_h_ = conv_param.stride(0);
      stride_w_ = conv_param.stride(0);
    } else {
      stride_h_ = conv_param.stride(0);
      stride_w_ = conv_param.stride(1);
    }
  }
  if (conv_param.has_pad_h() && conv_param.has_pad_w()) {
    pad_h_ = conv_param.pad_h();
    pad_w_ = conv_param.pad_w();
  } else {
    if (conv_param.pad_size() == 0) {
      pad_h_ = 0;
      pad_w_ = 0;
    }
    else if (conv_param.pad_size() == 1) {
      pad_h_ = conv_param.pad(0);
      pad_w_ = conv_param.pad(0);
    } else {
      pad_h_ = conv_param.pad(0);
      pad_w_ = conv_param.pad(1);
    }
  }
  if (conv_param.dilation_size() > 0) {
    if (conv_param.dilation_size() == 1) {
      dilation_h_ = conv_param.dilation(0);
      dilation_w_ = conv_param.dilation(0);
    } else {
      dilation_h_ = conv_param.dilation(0);
      dilation_w_ = conv_param.dilation(1);
    }
  } else {
    dilation_h_ = 1;
    dilation_w_ = 1;
  }
  vector<int> weight_shape(4);
  weight_shape[0] = bottom[0]->channels();
  weight_shape[1] = 1;
  weight_shape[2] = kernel_h_;
  weight_shape[3] = kernel_w_;
  vector<int> bias_shape;
  if (conv_param.bias_term()) {
    bias_shape.push_back(bottom[0]->channels());
  }
  if (this->blobs_.size() == 0) {
    if (conv_param.bias_term()) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    this->blobs_[0].reset(new Blob(weight_shape));
    shared_ptr<Filler> weight_filler(GetFiller(conv_param.weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    if (conv_param.bias_term()) {
      this->blobs_[1].reset(new Blob(bias_shape));
      shared_ptr<Filler> bias_filler(GetFiller(conv_param.bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
}

void ConvolutionDepthwiseLayer::Reshape(const vector<Blob*>& bottom,
                                        const vector<Blob*>& top) {
  vector<int> top_shape;
  top_shape.push_back(bottom[0]->num());
  top_shape.push_back(bottom[0]->channels());
  top_shape.push_back((bottom[0]->height() + 2 * pad_h_ - (dilation_h_ * (kernel_h_ - 1) + 1)) / stride_h_ + 1);
  top_shape.push_back((bottom[0]->width() + 2 * pad_w_ - (dilation_w_ * (kernel_w_ - 1) + 1)) / stride_w_ + 1);
  top[0]->Reshape(top_shape);
}

void ConvolutionDepthwiseLayer::Forward_cpu(const vector<Blob*>& bottom,
                                            const vector<Blob*>& top) {
  const int num = top[0]->num();
  const int channels = top[0]->channels();
  const int top_height = top[0]->height();
  const int top_width = top[0]->width();
  const int bottom_height = bottom[0]->height();
  const int bottom_width = bottom[0]->width();
  const real_t* bottom_data = bottom[0]->cpu_data();
  const real_t* weight_data_base = this->blobs_[0]->cpu_data();
  real_t* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < top_height; ++h) {
        for (int w = 0; w < top_width; ++w) {
          const real_t* weight_data = weight_data_base + c * kernel_h_ * kernel_w_;
          real_t value = 0;
          for (int kh = 0; kh < kernel_h_; ++kh) {
            for (int kw = 0; kw < kernel_w_; ++kw) {
              int h_in = -pad_h_ + h * stride_h_ + kh * dilation_h_;
              int w_in = -pad_w_ + w * stride_w_ + kw * dilation_w_;
              if ((h_in >= 0) && (h_in < bottom_height) && (w_in >= 0) && (w_in < bottom_width)) {
                int offset = ((n * channels + c) * bottom_height + h_in) * bottom_width + w_in;
                value += (*weight_data) * bottom_data[offset];
              }
              ++weight_data;
            }
          }
          *top_data++ = value;
        }
      }
    }
  }
  if (this->layer_param_.convolution_param().bias_term()) {
    top_data = top[0]->mutable_cpu_data();
    for (int n = 0; n < num; ++n) {
      const real_t* bias_data = this->blobs_[1]->cpu_data();
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < top_height; ++h) {
          for (int w = 0; w < top_width; ++w) {
            *top_data += *bias_data;
            ++top_data;
          }
        }
        ++bias_data;
      }
    }
  }
}

#ifndef USE_CUDA
STUB_GPU(ConvolutionDepthwiseLayer);
#endif

REGISTER_LAYER_CLASS(ConvolutionDepthwise);

}  // namespace caffe
