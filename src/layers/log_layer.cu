#include <vector>

#include "./log_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void LogLayer::Forward_gpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top) {
  const int count = bottom[0]->count();
  const real_t* bottom_data = bottom[0]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  if (input_scale_ == static_cast<real_t>(1) &&
      input_shift_ == static_cast<real_t>(0)) {
    caffe_gpu_log(count, bottom_data, top_data);
  } else {
    caffe_copy(count, bottom_data, top_data);
    if (input_scale_ != static_cast<real_t>(1)) {
      caffe_gpu_scal(count, input_scale_, top_data);
    }
    if (input_shift_ != static_cast<real_t>(0)) {
      caffe_gpu_add_scalar(count, input_shift_, top_data);
    }
    caffe_gpu_log(count, top_data, top_data);
  }
  if (base_scale_ != static_cast<real_t>(1)) {
    caffe_gpu_scal(count, base_scale_, top_data);
  }
}

}  // namespace caffe
