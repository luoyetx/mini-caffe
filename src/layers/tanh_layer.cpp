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
  TanHParameter_Engine engine = param.tanh_param().engine();
  if (engine == TanHParameter_Engine_DEFAULT) {
    engine = TanHParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = TanHParameter_Engine_CUDNN;
#endif
  }
  if (engine == TanHParameter_Engine_CAFFE) {
    return shared_ptr<Layer>(new TanHLayer(param));
#ifdef USE_CUDNN
  } else if (engine == TanHParameter_Engine_CUDNN) {
    return shared_ptr<Layer>(new CuDNNTanHLayer(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(TanH, CreateLayer);

}  // namespace caffe
