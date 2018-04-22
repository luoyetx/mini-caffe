#ifndef CAFFE_BASE_CONVOLUTION_LAYER_HPP_
#define CAFFE_BASE_CONVOLUTION_LAYER_HPP_

#include <vector>

#include "../layer.hpp"
#include "../util/im2col.hpp"

namespace caffe {

/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 */
class BaseConvolutionLayer : public Layer {
 public:
  explicit BaseConvolutionLayer(const LayerParameter& param)
      : Layer(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
                          const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
                       const vector<Blob*>& top);
  virtual vector<Blob*> GetTempBlobs() { return {&col_buffer_}; }

  virtual int MinBottomBlobs() const { return 1; }
  virtual int MinTopBlobs() const { return 1; }
  virtual bool EqualNumBottomTopBlobs() const { return true; }

 protected:
  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the im2col if
  // we just called weight_cpu_gemm with the same input.
  void forward_cpu_gemm(const real_t* input, const real_t* weights,
                        real_t* output, bool skip_im2col = false);
  void forward_cpu_bias(real_t* output, const real_t* bias);
  void backward_cpu_gemm(const real_t* input, const real_t* weights,
                         real_t* output);

#ifdef USE_CUDA
  void forward_gpu_gemm(const real_t* col_input, const real_t* weights,
                        real_t* output, bool skip_im2col = false);
  void forward_gpu_bias(real_t* output, const real_t* bias);
  void backward_gpu_gemm(const real_t* input, const real_t* weights,
                         real_t* col_output);
#endif

  /// @brief The spatial dimensions of the input.
  inline int input_shape(int i) {
    return (*bottom_shape_)[channel_axis_ + i];
  }
  // reverse_dimensions should return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  virtual bool reverse_dimensions() = 0;
  // Compute height_out_ and width_out_ from other parameters.
  virtual void compute_output_shape() = 0;

  /// @brief The spatial dimensions of a filter kernel.
  BlobInt kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  BlobInt stride_;
  /// @brief The spatial dimensions of the padding.
  BlobInt pad_;
  /// @brief The spatial dimensions of the dilation.
  BlobInt dilation_;
  /// @brief The spatial dimensions of the convolution input.
  BlobInt conv_input_shape_;
  /// @brief The spatial dimensions of the col_buffer.
  vector<int> col_buffer_shape_;
  /// @brief The spatial dimensions of the output.
  vector<int> output_shape_;
  const vector<int>* bottom_shape_;

  int num_spatial_axes_;
  int bottom_dim_;
  int top_dim_;

  int channel_axis_;
  int num_;
  int channels_;
  int group_;
  int out_spatial_dim_;
  int weight_offset_;
  int num_output_;
  bool bias_term_;
  bool is_1x1_;
  bool force_nd_im2col_;

 private:
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  inline void conv_im2col_cpu(const real_t* data, real_t* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_cpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), col_buff);
    }
  }
  inline void conv_col2im_cpu(const real_t* col_buff, real_t* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_cpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_cpu(col_buff, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), data);
    }
  }

#ifdef USE_CUDA
  inline void conv_im2col_gpu(const real_t* data, real_t* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_gpu(data, conv_in_channels_,
        conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
        kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
        pad_.cpu_data()[0], pad_.cpu_data()[1],
        stride_.cpu_data()[0], stride_.cpu_data()[1],
        dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
    }
    else {
      im2col_nd_gpu(data, num_spatial_axes_, num_kernels_im2col_,
        conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
        kernel_shape_.gpu_data(), pad_.gpu_data(),
        stride_.gpu_data(), dilation_.gpu_data(), col_buff);
    }
  }
  inline void conv_col2im_gpu(const real_t* col_buff, real_t* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_gpu(col_buff, conv_in_channels_,
        conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
        kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
        pad_.cpu_data()[0], pad_.cpu_data()[1],
        stride_.cpu_data()[0], stride_.cpu_data()[1],
        dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    }
    else {
      col2im_nd_gpu(col_buff, num_spatial_axes_, num_kernels_col2im_,
        conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
        kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
        dilation_.gpu_data(), data);
    }
  }

#endif  // USE_CUDA

  int num_kernels_im2col_;
  int num_kernels_col2im_;
  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int kernel_dim_;
  int col_offset_;
  int output_offset_;

  Blob col_buffer_;
  Blob bias_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_BASE_CONVOLUTION_LAYER_HPP_
