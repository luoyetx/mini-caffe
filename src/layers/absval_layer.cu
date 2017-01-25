#include <vector>

#include "./absval_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void AbsValLayer::Forward_gpu(const vector<Blob*>& bottom,
                              const vector<Blob*>& top) {
  const int count = top[0]->count();
  real_t* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_abs(count, bottom[0]->gpu_data(), top_data);
}

}  // namespace caffe
