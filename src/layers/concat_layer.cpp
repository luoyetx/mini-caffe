#include <vector>

#include "./concat_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void ConcatLayer::LayerSetUp(const vector<Blob*>& bottom,
                             const vector<Blob*>& top) {
  const ConcatParameter& concat_param = this->layer_param_.concat_param();
  CHECK(!(concat_param.has_axis() && concat_param.has_concat_dim()))
      << "Either axis or concat_dim should be specified; not both.";
}

void ConcatLayer::Reshape(const vector<Blob*>& bottom,
                          const vector<Blob*>& top) {
  const int num_axes = bottom[0]->num_axes();
  const ConcatParameter& concat_param = this->layer_param_.concat_param();
  if (concat_param.has_concat_dim()) {
    concat_axis_ = static_cast<int>(concat_param.concat_dim());
    // Don't allow negative indexing for concat_dim, a uint32 -- almost
    // certainly unintended.
    CHECK_GE(concat_axis_, 0) << "casting concat_dim from uint32 to int32 "
        << "produced negative result; concat_dim must satisfy "
        << "0 <= concat_dim < " << kMaxBlobAxes;
    CHECK_LT(concat_axis_, num_axes) << "concat_dim out of range.";
  } else {
    concat_axis_ = bottom[0]->CanonicalAxisIndex(concat_param.axis());
  }
  // Initialize with the first blob.
  vector<int> top_shape = bottom[0]->shape();
  num_concats_ = bottom[0]->count(0, concat_axis_);
  concat_input_size_ = bottom[0]->count(concat_axis_ + 1);
  int bottom_count_sum = bottom[0]->count();
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK_EQ(num_axes, bottom[i]->num_axes())
        << "All inputs must have the same #axes.";
    for (int j = 0; j < num_axes; ++j) {
      if (j == concat_axis_) { continue; }
      CHECK_EQ(top_shape[j], bottom[i]->shape(j))
          << "All inputs must have the same shape, except at concat_axis.";
    }
    bottom_count_sum += bottom[i]->count();
    top_shape[concat_axis_] += bottom[i]->shape(concat_axis_);
  }
  top[0]->Reshape(top_shape);
  CHECK_EQ(bottom_count_sum, top[0]->count());
  if (bottom.size() == 1) {
    top[0]->ShareData(*bottom[0]);
  }
}

void ConcatLayer::Forward_cpu(const vector<Blob*>& bottom,
                              const vector<Blob*>& top) {
  if (bottom.size() == 1) { return; }
  real_t* top_data = top[0]->mutable_cpu_data();
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  for (int i = 0; i < bottom.size(); ++i) {
    const real_t* bottom_data = bottom[i]->cpu_data();
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    for (int n = 0; n < num_concats_; ++n) {
      caffe_copy(bottom_concat_axis * concat_input_size_,
        bottom_data + n * bottom_concat_axis * concat_input_size_,
        top_data + (n * top_concat_axis + offset_concat_axis)
            * concat_input_size_);
    }
    offset_concat_axis += bottom_concat_axis;
  }
}

#ifndef USE_CUDA
STUB_GPU(ConcatLayer);
#endif

REGISTER_LAYER_CLASS(Concat);

}  // namespace caffe
