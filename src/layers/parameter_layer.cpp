#include "./parameter_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void ParameterLayer::Forward_cpu(const vector<Blob*>& bottom,
                                 const vector<Blob*>& top) {
  caffe_copy(blobs_[0]->count(), blobs_[0]->cpu_data(), top[0]->mutable_cpu_data());
}

void ParameterLayer::Forward_gpu(const vector<Blob*>& bottom,
                                 const vector<Blob*>& top) {
  caffe_copy(blobs_[0]->count(), blobs_[0]->gpu_data(), top[0]->mutable_gpu_data());
}


REGISTER_LAYER_CLASS(Parameter);

}  // namespace caffe
