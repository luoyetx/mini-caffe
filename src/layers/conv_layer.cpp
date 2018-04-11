#include <vector>

#include "./conv_layer.hpp"
#include "./conv_dw_layer.hpp"

#ifdef USE_CUDNN
#include "./cudnn/cudnn_conv_layer.hpp"
#endif  // USE_CUDNN

namespace caffe {

void ConvolutionLayer::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent) / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

void ConvolutionLayer::Forward_cpu(const vector<Blob*>& bottom,
                                   const vector<Blob*>& top) {
  const real_t* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const real_t* bottom_data = bottom[i]->cpu_data();
    real_t* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const real_t* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

#ifndef USE_CUDA
STUB_GPU(ConvolutionLayer);
#endif

// Creator

static shared_ptr<Layer> CreateLayer(const LayerParameter &param) {
#ifdef USE_CUDNN
  if (Caffe::mode() == Caffe::GPU) {
    ConvolutionParameter conv_param = param.convolution_param();
    if (conv_param.group() == conv_param.num_output()) {  // depthwise
      return shared_ptr<Layer>(new ConvolutionDepthwiseLayer(param));
    }
    bool use_dilation = false;
    for (int i = 0; i < conv_param.dilation_size(); ++i) {
      if (conv_param.dilation(i) > 1) {
        use_dilation = true;
      }
    }
    if (!use_dilation) {
      return shared_ptr<Layer>(new CuDNNConvolutionLayer(param));
    }
  }
#endif  // USE_CUDNN
  return shared_ptr<Layer>(new ConvolutionLayer(param));
}

REGISTER_LAYER_CREATOR(Convolution, CreateLayer);

}  // namespace caffe
