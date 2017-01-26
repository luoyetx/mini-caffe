#include <cmath>
#include <vector>

#include "./sigmoid_layer.hpp"

namespace caffe {

inline real_t sigmoid(real_t x) {
  return 1. / (1. + exp(-x));
}

void SigmoidLayer::Forward_cpu(const vector<Blob*>& bottom,
                               const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->cpu_data();
  real_t* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = sigmoid(bottom_data[i]);
  }
}

#ifndef USE_CUDA
STUB_GPU(SigmoidLayer);
#endif

}  // namespace caffe
