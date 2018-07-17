#include <vector>

#include "./flatten_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void FlattenLayer::Reshape(const vector<Blob*>& bottom,
                           const vector<Blob*>& top) {
  const int start_axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.flatten_param().axis());
  const int end_axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.flatten_param().end_axis());
  vector<int> top_shape;
  for (int i = 0; i < start_axis; ++i) {
    top_shape.push_back(bottom[0]->shape(i));
  }
  const int flattened_dim = bottom[0]->count(start_axis, end_axis + 1);
  top_shape.push_back(flattened_dim);
  for (int i = end_axis + 1; i < bottom[0]->num_axes(); ++i) {
    top_shape.push_back(bottom[0]->shape(i));
  }
  top[0]->Reshape(top_shape);
  CHECK_EQ(top[0]->count(), bottom[0]->count());
}

void FlattenLayer::Forward_cpu(const vector<Blob*>& bottom,
                               const vector<Blob*>& top) {
  if (bottom[0] != top[0]) {
    caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(), top[0]->mutable_cpu_data());
  }
}

void FlattenLayer::Forward_gpu(const vector<Blob*>& bottom,
  const vector<Blob*>& top) {
  if (bottom[0] != top[0]) {
    caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
  }
}

REGISTER_LAYER_CLASS(Flatten);

}  // namespace caffe
