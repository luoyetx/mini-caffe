#include <vector>

#include "./threshold_layer.hpp"

namespace caffe {

void ThresholdLayer::LayerSetUp(const vector<Blob*>& bottom,
                                const vector<Blob*>& top) {
  NeuronLayer::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.threshold_param().threshold();
}

void ThresholdLayer::Forward_cpu(const vector<Blob*>& bottom,
                                 const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->cpu_data();
  real_t* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = (bottom_data[i] > threshold_) ? 1 : 0;
  }
}

#ifndef USE_CUDA
STUB_GPU(ThresholdLayer);
#endif

REGISTER_LAYER_CLASS(Threshold);

}  // namespace caffe
