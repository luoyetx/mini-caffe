#include <vector>

#include "./filter_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void FilterLayer::Forward_gpu(const vector<Blob*>& bottom,
                              const vector<Blob*>& top) {
  int new_tops_num = indices_to_forward_.size();
  // forward all filtered items for all bottoms but the Selector (bottom[last])
  for (int t = 0; t < top.size(); ++t) {
    const real_t* bottom_data = bottom[t]->gpu_data();
    real_t* top_data = top[t]->mutable_gpu_data();
    int dim = bottom[t]->count() / bottom[t]->shape(0);
    for (int n = 0; n < new_tops_num; ++n) {
      int data_offset_top = n * dim;
      int data_offset_bottom = indices_to_forward_[n] * dim;
      caffe_copy(dim, bottom_data + data_offset_bottom,
        top_data + data_offset_top);
    }
  }
}

}  // namespace caffe
