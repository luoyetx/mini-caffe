#include <vector>

#include "./exp_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void ExpLayer::LayerSetUp(const vector<Blob*>& bottom,
                          const vector<Blob*>& top) {
  NeuronLayer::LayerSetUp(bottom, top);
  const real_t base = this->layer_param_.exp_param().base();
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
  const real_t input_scale = this->layer_param_.exp_param().scale();
  const real_t input_shift = this->layer_param_.exp_param().shift();
  inner_scale_ = log_base * input_scale;
  outer_scale_ = (input_shift == static_cast<real_t>(0)) ? 1 :
     ( (base != static_cast<real_t>(-1)) ? pow(base, input_shift) : exp(input_shift) );
}

void ExpLayer::Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top) {
  const int count = bottom[0]->count();
  const real_t* bottom_data = bottom[0]->cpu_data();
  real_t* top_data = top[0]->mutable_cpu_data();
  if (inner_scale_ == static_cast<real_t>(1)) {
    caffe_exp(count, bottom_data, top_data);
  } else {
    caffe_cpu_scale(count, inner_scale_, bottom_data, top_data);
    caffe_exp(count, top_data, top_data);
  }
  if (outer_scale_ != static_cast<real_t>(1)) {
    caffe_scal(count, outer_scale_, top_data);
  }
}

#ifndef USE_CUDA
STUB_GPU(ExpLayer);
#endif

REGISTER_LAYER_CLASS(Exp);

}  // namespace caffe
