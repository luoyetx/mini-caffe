#include <algorithm>
#include <vector>

#include "./batch_norm_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void BatchNormLayer::LayerSetUp(const vector<Blob*>& bottom,
                                const vector<Blob*>& top) {
  BatchNormParameter param = this->layer_param_.batch_norm_param();
  use_global_stats_ = true;
  if (param.has_use_global_stats())
    use_global_stats_ = param.use_global_stats();
  if (bottom[0]->num_axes() == 1)
    channels_ = 1;
  else
    channels_ = bottom[0]->shape(1);
  eps_ = param.eps();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(3);
    vector<int> sz;
    sz.push_back(channels_);
    this->blobs_[0].reset(new Blob(sz));
    this->blobs_[1].reset(new Blob(sz));
    sz[0]=1;
    this->blobs_[2].reset(new Blob(sz));
    for (int i = 0; i < 3; ++i) {
      caffe_set(this->blobs_[i]->count(), static_cast<real_t>(0),
                this->blobs_[i]->mutable_cpu_data());
    }
  }
  // set temp blob name
  temp_.set_name(this->layer_param_.name() + "__temp__");
}

void BatchNormLayer::Reshape(const vector<Blob*>& bottom,
                             const vector<Blob*>& top) {
  if (bottom[0]->num_axes() >= 1)
    CHECK_EQ(bottom[0]->shape(1), channels_);
  top[0]->ReshapeLike(*bottom[0]);

  vector<int> sz;
  sz.push_back(channels_);
  mean_.Reshape(sz);
  variance_.Reshape(sz);
  temp_.ReshapeLike(*bottom[0]);
  sz[0] = bottom[0]->shape(0);
  batch_sum_multiplier_.Reshape(sz);

  int spatial_dim = bottom[0]->count()/(channels_*bottom[0]->shape(0));
  if (spatial_sum_multiplier_.num_axes() == 0 ||
      spatial_sum_multiplier_.shape(0) != spatial_dim) {
    sz[0] = spatial_dim;
    spatial_sum_multiplier_.Reshape(sz);
    real_t* multiplier_data = spatial_sum_multiplier_.mutable_cpu_data();
    caffe_set(spatial_sum_multiplier_.count(), static_cast<real_t>(1), multiplier_data);
  }

  int numbychans = channels_*bottom[0]->shape(0);
  if (num_by_chans_.num_axes() == 0 ||
      num_by_chans_.shape(0) != numbychans) {
    sz[0] = numbychans;
    num_by_chans_.Reshape(sz);
    caffe_set(batch_sum_multiplier_.count(), static_cast<real_t>(1),
      batch_sum_multiplier_.mutable_cpu_data());
  }
}

void BatchNormLayer::Forward_cpu(const vector<Blob*>& bottom,
                                 const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->cpu_data();
  real_t* top_data = top[0]->mutable_cpu_data();
  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);

  if (bottom[0] != top[0]) {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }

  if (use_global_stats_) {
    // use the stored mean/variance estimates.
    const real_t scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
        0 : 1 / this->blobs_[2]->cpu_data()[0];
    caffe_cpu_scale(variance_.count(), scale_factor,
        this->blobs_[0]->cpu_data(), mean_.mutable_cpu_data());
    caffe_cpu_scale(variance_.count(), scale_factor,
        this->blobs_[1]->cpu_data(), variance_.mutable_cpu_data());
  } else {
    // compute mean
    caffe_cpu_gemv(CblasNoTrans, channels_ * num, spatial_dim,
        1. / (num * spatial_dim), bottom_data,
        spatial_sum_multiplier_.cpu_data(), static_cast<real_t>(0),
        num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemv(CblasTrans, num, channels_, static_cast<real_t>(1),
        num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), static_cast<real_t>(0),
        mean_.mutable_cpu_data());
  }

  // subtract mean
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), static_cast<real_t>(0),
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, -1, num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), static_cast<real_t>(1), top_data);

  if (!use_global_stats_) {
    // compute variance using var(X) = E((X-EX)^2)
    caffe_powx(top[0]->count(), top_data, static_cast<real_t>(2),
        temp_.mutable_cpu_data());  // (X-EX)^2
    caffe_cpu_gemv(CblasNoTrans, channels_ * num, spatial_dim,
        1. / (num * spatial_dim), temp_.cpu_data(),
        spatial_sum_multiplier_.cpu_data(), static_cast<real_t>(0),
        num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemv(CblasTrans, num, channels_, static_cast<real_t>(1),
        num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), static_cast<real_t>(0),
        variance_.mutable_cpu_data());  // E((X_EX)^2)
  }

  // normalize variance
  caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
  caffe_powx(variance_.count(), variance_.cpu_data(), static_cast<real_t>(0.5),
             variance_.mutable_cpu_data());

  // replicate variance to input size
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), variance_.cpu_data(), static_cast<real_t>(0),
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, static_cast<real_t>(1), num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), static_cast<real_t>(0), temp_.mutable_cpu_data());
  caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
}

#ifndef USE_CUDA
STUB_GPU(BatchNormLayer);
#endif

REGISTER_LAYER_CLASS(BatchNorm);

}  // namespace caffe
