#ifdef USE_NNPACK

#include "./nnpack_pooling_layer.hpp"

namespace caffe {

void NNPackPoolingLayer::Forward_cpu(const vector<Blob*>& bottom,
                                     const vector<Blob*>& top) {
  CHECK_EQ(this->layer_param_.pooling_param().pool(),
           PoolingParameter_PoolMethod_MAX);
  if (this->kernel_w_ != 2 || this->kernel_h_ != 2 ||
      this->stride_w_ != 2 || this->stride_h_ != 2 ||
      this->pad_w_ != 2 || this->pad_h_ != 2) {
    return PoolingLayer::Forward_cpu(bottom, top);
  }
  nnp_size input_size = {static_cast<size_t>(bottom[0]->width()),
                         static_cast<size_t>(bottom[0]->height())};
  nnp_padding input_padding = {static_cast<size_t>(this->pad_h_),
                               static_cast<size_t>(this->pad_w_),
                               static_cast<size_t>(this->pad_h_),
                               static_cast<size_t>(this->pad_w_)};
  nnp_size pooling_size = {static_cast<size_t>(this->kernel_w_),
                           static_cast<size_t>(this->kernel_h_)};
  nnp_size pooling_stride = {static_cast<size_t>(this->stride_w_),
                             static_cast<size_t>(this->stride_h_)};
  auto status = nnp_max_pooling_output(
      bottom[0]->num(),
      bottom[0]->channels(),
      input_size,
      input_padding,
      pooling_size,
      pooling_stride,
      bottom[0]->cpu_data(),
      top[0]->mutable_cpu_data(),
      NNPack::Get().threadpool());
  CHECK_EQ(status, nnp_status_success);
}

}  // namespace caffe

#endif  // USE_NNPACK
