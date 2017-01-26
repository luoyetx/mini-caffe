#include <vector>

#include "./dropout_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void DropoutLayer::Forward_gpu(const vector<Blob*>& bottom,
                               const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  caffe_copy(count, bottom_data, top_data);
}

}  // namespace caffe
