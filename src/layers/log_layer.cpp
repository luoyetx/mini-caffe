#include <vector>

#include "./log_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void LogLayer::LayerSetUp(const vector<Blob*>& bottom,
                          const vector<Blob*>& top) {
  NeuronLayer::LayerSetUp(bottom, top);
  const real_t base = this->layer_param_.log_param().base();
  if (base != static_cast<real_t>(-1)) {
    CHECK_GT(base, 0) << "base must be strictly positive.";
  }
  // If base == -1, interpret the base as e and set log_base = 1 exactly.
  // Otherwise, calculate its log explicitly.
  const real_t log_base = (base == static_cast<real_t>(-1)) ? 1 : log(base);
  CHECK(!std::isnan(log_base))
      << "NaN result: log(base) = log(" << base << ") = " << log_base;
  CHECK(!std::isinf(log_base))
      << "Inf result: log(base) = log(" << base << ") = " << log_base;
  base_scale_ = static_cast<real_t>(1) / log_base;
  CHECK(!std::isnan(base_scale_))
      << "NaN result: 1/log(base) = 1/log(" << base << ") = " << base_scale_;
  CHECK(!std::isinf(base_scale_))
      << "Inf result: 1/log(base) = 1/log(" << base << ") = " << base_scale_;
  input_scale_ = this->layer_param_.log_param().scale();
  input_shift_ = this->layer_param_.log_param().shift();
  backward_num_scale_ = input_scale_ / log_base;
}

void LogLayer::Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top) {
  const int count = bottom[0]->count();
  const real_t* bottom_data = bottom[0]->cpu_data();
  real_t* top_data = top[0]->mutable_cpu_data();
  if (input_scale_ == static_cast<real_t>(1) && input_shift_ == static_cast<real_t>(0)) {
    caffe_log(count, bottom_data, top_data);
  } else {
    caffe_copy(count, bottom_data, top_data);
    if (input_scale_ != static_cast<real_t>(1)) {
      caffe_scal(count, input_scale_, top_data);
    }
    if (input_shift_ != static_cast<real_t>(0)) {
      caffe_add_scalar(count, input_shift_, top_data);
    }
    caffe_log(count, top_data, top_data);
  }
  if (base_scale_ != static_cast<real_t>(1)) {
    caffe_scal(count, base_scale_, top_data);
  }
}

#ifndef USE_CUDA
STUB_GPU(LogLayer);
#endif

REGISTER_LAYER_CLASS(Log);

}  // namespace caffe
