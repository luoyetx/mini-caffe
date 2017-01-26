#include <vector>

#include "./neuron_layer.hpp"

namespace caffe {

void NeuronLayer::Reshape(const vector<Blob*>& bottom,
                          const vector<Blob*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

}  // namespace caffe
