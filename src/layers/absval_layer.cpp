#include <vector>

#include "./absval_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AbsValLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
}

template <typename Dtype>
void AbsValLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_abs(count, bottom[0]->cpu_data(), top_data);
}

#ifndef USE_CUDA
STUB_GPU(AbsValLayer);
#endif

INSTANTIATE_CLASS(AbsValLayer);
REGISTER_LAYER_CLASS(AbsVal);

}  // namespace caffe
