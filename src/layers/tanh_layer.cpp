// TanH neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <vector>

#include "./tanh_layer.hpp"

#ifdef USE_CUDNN
#include "./cudnn/cudnn_tanh_layer.hpp"
#endif  // USE_CUDNN

namespace caffe {

void TanHLayer::Forward_cpu(const vector<Blob*>& bottom,
                            const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->cpu_data();
  real_t* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = tanh(bottom_data[i]);
  }
}

#ifndef USE_CUDA
STUB_GPU(TanHLayer);
#endif

// Creator

static shared_ptr<Layer> CreateLayer(const LayerParameter& param) {
#ifdef USE_CUDNN
  if (Caffe::mode() == Caffe::GPU) {
    return shared_ptr<Layer>(new CuDNNTanHLayer(param));
  }
#endif  // USE_CUDNN
  return shared_ptr<Layer>(new TanHLayer(param));
}

REGISTER_LAYER_CREATOR(TanH, CreateLayer);

}  // namespace caffe
