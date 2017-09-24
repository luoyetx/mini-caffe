#include <algorithm>
#include <vector>

#include "./softmax_layer.hpp"
#include "../util/math_functions.hpp"

#ifdef USE_CUDNN
#include "./cudnn/cudnn_softmax_layer.hpp"
#endif  // USE_CUDNN

namespace caffe {

void SoftmaxLayer::Reshape(const vector<Blob*>& bottom,
                           const vector<Blob*>& top) {
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  top[0]->ReshapeLike(*bottom[0]);
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));
  sum_multiplier_.Reshape(mult_dims);
  real_t* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), static_cast<real_t>(1), multiplier_data);
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  vector<int> scale_dims = bottom[0]->shape();
  scale_dims[softmax_axis_] = 1;
  scale_.Reshape(scale_dims);
}

void SoftmaxLayer::Forward_cpu(const vector<Blob*>& bottom,
                               const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->cpu_data();
  real_t* top_data = top[0]->mutable_cpu_data();
  real_t* scale_data = scale_.mutable_cpu_data();
  int channels = bottom[0]->shape(softmax_axis_);
  int dim = bottom[0]->count() / outer_num_;
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  for (int i = 0; i < outer_num_; ++i) {
    // initialize scale_data to the first plane
    caffe_copy(inner_num_, bottom_data + i * dim, scale_data);
    for (int j = 0; j < channels; j++) {
      for (int k = 0; k < inner_num_; k++) {
        scale_data[k] = std::max(scale_data[k],
            bottom_data[i * dim + j * inner_num_ + k]);
      }
    }
    // subtraction
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, channels, inner_num_,
      1, static_cast<real_t>(-1), sum_multiplier_.cpu_data(), scale_data,
      static_cast<real_t>(1), top_data);
    // exponentiation
    caffe_exp(dim, top_data, top_data);
    // sum after exp
    caffe_cpu_gemv(CblasTrans, channels, inner_num_, static_cast<real_t>(1),
      top_data, sum_multiplier_.cpu_data(), static_cast<real_t>(0), scale_data);
    // division
    for (int j = 0; j < channels; j++) {
      caffe_div(inner_num_, top_data, scale_data, top_data);
      top_data += inner_num_;
    }
  }
}

#ifndef USE_CUDA
STUB_GPU(SoftmaxLayer);
#endif

// Creator

static shared_ptr<Layer> CreateLayer(const LayerParameter& param) {
#ifdef USE_CUDNN
  if (Caffe::mode() == Caffe::GPU) {
    return shared_ptr<Layer>(new CuDNNSoftmaxLayer(param));
  }
#endif  // USE_CUDNN
  return shared_ptr<Layer>(new SoftmaxLayer(param));
}

REGISTER_LAYER_CREATOR(Softmax, CreateLayer);

}  // namespace caffe
