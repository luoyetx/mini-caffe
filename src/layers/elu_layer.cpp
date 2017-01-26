#include <algorithm>
#include <vector>

#include "./elu_layer.hpp"

namespace caffe {

void ELULayer::Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->cpu_data();
  real_t* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  real_t alpha = this->layer_param_.elu_param().alpha();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], static_cast<real_t>(0))
        + alpha * (exp(std::min(bottom_data[i], static_cast<real_t>(0))) - 1);
  }
}

#ifndef USE_CUDA
STUB_GPU(ELULayer);
#endif

REGISTER_LAYER_CLASS(ELU);

}  // namespace caffe
