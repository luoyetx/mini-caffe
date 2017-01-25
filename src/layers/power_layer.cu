#include <vector>

#include "./power_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void PowerLayer::Forward_gpu(const vector<Blob*>& bottom,
                             const vector<Blob*>& top) {
  real_t* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // Special case where we can ignore the input: scale or power is 0.
  if (diff_scale_ == static_cast<real_t>(0)) {
    real_t value = (power_ == 0) ? 1 : pow(shift_, power_);
    caffe_gpu_set(count, value, top_data);
    return;
  }
  const real_t* bottom_data = bottom[0]->gpu_data();
  caffe_copy(count, bottom_data, top_data);
  if (scale_ != static_cast<real_t>(1)) {
    caffe_gpu_scal(count, scale_, top_data);
  }
  if (shift_ != static_cast<real_t>(0)) {
    caffe_gpu_add_scalar(count, shift_, top_data);
  }
  if (power_ != static_cast<real_t>(1)) {
    caffe_gpu_powx(count, top_data, power_, top_data);
  }
}

}  // namespace caffe
