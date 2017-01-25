#include <vector>

#include "./exp_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void ExpLayer::Forward_gpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top) {
  const int count = bottom[0]->count();
  const real_t* bottom_data = bottom[0]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  if (inner_scale_ == static_cast<real_t>(1)) {
    caffe_gpu_exp(count, bottom_data, top_data);
  } else {
    caffe_gpu_scale(count, inner_scale_, bottom_data, top_data);
    caffe_gpu_exp(count, top_data, top_data);
  }
  if (outer_scale_ != static_cast<real_t>(1)) {
    caffe_gpu_scal(count, outer_scale_, top_data);
  }
}

}  // namespace caffe
