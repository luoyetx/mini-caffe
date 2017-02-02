#ifdef USE_NNPACK

#include "./nnpack_conv_layer.hpp"

namespace caffe {

void NNPackConvolutionLayer::Forward_cpu(const vector<Blob*>& bottom,
                                         const vector<Blob*>& top) {
  CHECK_EQ(this->group_, 1);
  CHECK_EQ(this->bias_term_, true);
  CHECK_EQ(this->blobs_.size(), 2);

  bool is_stride_1 = true;
  for (auto i = 0; i < num_spatial_axes_; ++i) {
    if (this->stride_.cpu_data()[i] != 1) {
      is_stride_1 = false;
    }
  }
  if (num_spatial_axes_ != 2 || !is_stride_1) {
    return ConvolutionLayer::Forward_cpu(bottom, top);
  }

  shared_ptr<Blob> weight = this->blobs_[0];
  shared_ptr<Blob> bias = this->blobs_[1];
  for (int i = 0; i < bottom.size(); ++i) {
    const size_t input_c = bottom[i]->channels();
    const size_t output_c = top[i]->channels();
    nnp_size input_size = {static_cast<size_t>(bottom[i]->width()),
                           static_cast<size_t>(bottom[i]->height())};
    nnp_padding input_padding = {static_cast<size_t>(this->pad_.cpu_data()[0]),
                                 static_cast<size_t>(this->pad_.cpu_data()[1]),
                                 static_cast<size_t>(this->pad_.cpu_data()[0]),
                                 static_cast<size_t>(this->pad_.cpu_data()[1])};
    nnp_size kernel_size = {static_cast<size_t>(weight->width()),
                            static_cast<size_t>(weight->height())};
    nnp_size output_subsampling = {static_cast<size_t>(this->stride_.cpu_data()[1]),
                                   static_cast<size_t>(this->stride_.cpu_data()[0])};
    nnp_convolution_algorithm algorithm = nnp_convolution_algorithm_auto;
    nnp_convolution_transform_strategy kts = nnp_convolution_transform_strategy_tuple_based;
    nnp_status status = nnp_status_success;
    if (bottom[i]->num() == 1) {
      status = nnp_convolution_inference(
          algorithm,
          kts,
          input_c,
          output_c,
          input_size,
          input_padding,
          kernel_size,
          output_subsampling,
          bottom[i]->cpu_data(),
          weight->cpu_data(),
          bias->cpu_data(),
          top[i]->mutable_cpu_data(),
          NNPack::Get().threadpool(),
          nullptr);
    }
    else {
      status = nnp_convolution_output(
          algorithm,
          bottom[i]->num(),
          input_c,
          output_c,
          input_size,
          input_padding,
          kernel_size,
          bottom[i]->cpu_data(),
          weight->cpu_data(),
          bias->cpu_data(),
          top[i]->mutable_cpu_data(),
          NNPack::Get().threadpool(),
          nullptr);
    }
    CHECK_EQ(status, nnp_status_success) << "nnpack convolution feedforward failed";
  }
}

} // namespace caffe

#endif  // USE_NNPACK
