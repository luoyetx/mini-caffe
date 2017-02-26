#include <algorithm>
#include <vector>

#include "./scale_layer.hpp"
#include "../filler.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void ScaleLayer::LayerSetUp(const vector<Blob*>& bottom,
                            const vector<Blob*>& top) {
  const ScaleParameter& param = this->layer_param_.scale_param();
  if (bottom.size() == 1 && this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else if (bottom.size() == 1) {
    // scale is a learned parameter; initialize it
    axis_ = bottom[0]->CanonicalAxisIndex(param.axis());
    const int num_axes = param.num_axes();
    CHECK_GE(num_axes, -1) << "num_axes must be non-negative, "
                           << "or -1 to extend to the end of bottom[0]";
    if (num_axes >= 0) {
      CHECK_GE(bottom[0]->num_axes(), axis_ + num_axes)
          << "scale blob's shape extends past bottom[0]'s shape when applied "
          << "starting with bottom[0] axis = " << axis_;
    }
    this->blobs_.resize(1);
    const vector<int>::const_iterator& shape_start =
        bottom[0]->shape().begin() + axis_;
    const vector<int>::const_iterator& shape_end =
        (num_axes == -1) ? bottom[0]->shape().end() : (shape_start + num_axes);
    vector<int> scale_shape(shape_start, shape_end);
    this->blobs_[0].reset(new Blob(scale_shape));
    FillerParameter filler_param(param.filler());
    if (!param.has_filler()) {
      // Default to unit (1) filler for identity operation.
      filler_param.set_type("constant");
      filler_param.set_value(1);
    }
    shared_ptr<Filler> filler(GetFiller(filler_param));
    filler->Fill(this->blobs_[0].get());
  }
  if (param.bias_term()) {
    LayerParameter layer_param(this->layer_param_);
    layer_param.set_type("Bias");
    BiasParameter* bias_param = layer_param.mutable_bias_param();
    bias_param->set_axis(param.axis());
    if (bottom.size() > 1) {
      bias_param->set_num_axes(bottom[1]->num_axes());
    } else {
      bias_param->set_num_axes(param.num_axes());
    }
    bias_param->mutable_filler()->CopyFrom(param.bias_filler());
    bias_layer_ = LayerRegistry::CreateLayer(layer_param);
    bias_bottom_vec_.resize(1);
    bias_bottom_vec_[0] = bottom[0];
    bias_layer_->SetUp(bias_bottom_vec_, top);
    bias_param_id_ = this->blobs_.size();
    this->blobs_.resize(bias_param_id_ + 1);
    this->blobs_[bias_param_id_] = bias_layer_->blobs()[0];
    bias_propagate_down_.resize(1, false);
  }
}

void ScaleLayer::Reshape(const vector<Blob*>& bottom,
                         const vector<Blob*>& top) {
  const ScaleParameter& param = this->layer_param_.scale_param();
  Blob* scale = (bottom.size() > 1) ? bottom[1] : this->blobs_[0].get();
  // Always set axis_ == 0 in special case where scale is a scalar
  // (num_axes == 0). Mathematically equivalent for any choice of axis_, so the
  // actual setting can be safely ignored; and computation is most efficient
  // with axis_ == 0 and (therefore) outer_dim_ == 1. (Setting axis_ to
  // bottom[0]->num_axes() - 1, giving inner_dim_ == 1, would be equally
  // performant.)
  axis_ = (scale->num_axes() == 0) ?
      0 : bottom[0]->CanonicalAxisIndex(param.axis());
  CHECK_GE(bottom[0]->num_axes(), axis_ + scale->num_axes())
      << "scale blob's shape extends past bottom[0]'s shape when applied "
      << "starting with bottom[0] axis = " << axis_;
  for (int i = 0; i < scale->num_axes(); ++i) {
    CHECK_EQ(bottom[0]->shape(axis_ + i), scale->shape(i))
        << "dimension mismatch between bottom[0]->shape(" << axis_ + i
        << ") and scale->shape(" << i << ")";
  }
  outer_dim_ = bottom[0]->count(0, axis_);
  scale_dim_ = scale->count();
  inner_dim_ = bottom[0]->count(axis_ + scale->num_axes());
  if (bottom[0] != top[0]) {
    top[0]->ReshapeLike(*bottom[0]);
  }
  if (bias_layer_) {
    bias_bottom_vec_[0] = top[0];
    bias_layer_->Reshape(bias_bottom_vec_, top);
  }
}

void ScaleLayer::Forward_cpu(const vector<Blob*>& bottom,
                             const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->cpu_data();
  const real_t* scale_data =
      ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->cpu_data();
  real_t* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < outer_dim_; ++n) {
    for (int d = 0; d < scale_dim_; ++d) {
      const real_t factor = scale_data[d];
      caffe_cpu_scale(inner_dim_, factor, bottom_data, top_data);
      bottom_data += inner_dim_;
      top_data += inner_dim_;
    }
  }
  if (bias_layer_) {
    bias_layer_->Forward(bias_bottom_vec_, top);
  }
}

#ifndef USE_CUDA
STUB_GPU(ScaleLayer);
#endif

REGISTER_LAYER_CLASS(Scale);

}  // namespace caffe
