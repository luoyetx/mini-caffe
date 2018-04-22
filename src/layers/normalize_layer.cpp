#include <vector>

#include "../filler.hpp"
#include "./normalize_layer.hpp"

namespace caffe {

void NormalizeLayer::LayerSetUp(const vector<Blob*>& bottom,
                                const vector<Blob*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  buffer_.Reshape(1, bottom[0]->channels(),
                   bottom[0]->height(), bottom[0]->width());
  buffer_channel_.Reshape(1, bottom[0]->channels(), 1, 1);
  buffer_spatial_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
  NormalizeParameter norm_param = this->layer_param().norm_param();
  across_spatial_ = norm_param.across_spatial();
  if (across_spatial_) {
    norm_.Reshape(bottom[0]->num(), 1, 1, 1);
  } else {
    norm_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  }
  eps_ = norm_param.eps();
  int channels = bottom[0]->channels();
  int spatial_dim = bottom[0]->width() * bottom[0]->height();
  sum_channel_multiplier_.Reshape(1, channels, 1, 1);
  caffe_set(channels, static_cast<real_t>(1), sum_channel_multiplier_.mutable_cpu_data());
  sum_spatial_multiplier_.Reshape(
      1, 1, bottom[0]->height(), bottom[0]->width());
  caffe_set(spatial_dim, static_cast<real_t>(1), sum_spatial_multiplier_.mutable_cpu_data());
  channel_shared_ = norm_param.channel_shared();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    if (channel_shared_) {
      this->blobs_[0].reset(new Blob(vector<int>(0)));
    } else {
      this->blobs_[0].reset(new Blob(vector<int>(1, channels)));
    }
    shared_ptr<Filler> scale_filler;
    if (norm_param.has_scale_filler()) {
      scale_filler.reset(GetFiller(norm_param.scale_filler()));
    } else {
      FillerParameter filler_param;
      filler_param.set_type("constant");
      filler_param.set_value(1.0);
      scale_filler.reset(GetFiller(filler_param));
    }
    scale_filler->Fill(this->blobs_[0].get());
  }
  if (channel_shared_) {
    CHECK_EQ(this->blobs_[0]->count(), 1)
        << "Scale size is inconsistent with prototxt config";
  } else {
    CHECK_EQ(this->blobs_[0]->count(), channels)
        << "Scale size is inconsistent with prototxt config";
  }
  // set temp blob name
  buffer_.set_name(this->layer_param_.name() + "__buffer__");
  buffer_spatial_.set_name(this->layer_param_.name() + "__buffer_spatial__");
  norm_.set_name(this->layer_param_.name() + "__norm__");
  sum_spatial_multiplier_.set_name(this->layer_param_.name() + "__sum_spatial_multiplier__");
}

void NormalizeLayer::Reshape(const vector<Blob*>& bottom,
                             const vector<Blob*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  top[0]->ReshapeLike(*bottom[0]);
  buffer_.Reshape(1, bottom[0]->channels(),
                   bottom[0]->height(), bottom[0]->width());
  if (!across_spatial_) {
    norm_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  }
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  if (spatial_dim != sum_spatial_multiplier_.count()) {
    sum_spatial_multiplier_.Reshape(
        1, 1, bottom[0]->height(), bottom[0]->width());
    caffe_set(spatial_dim, static_cast<real_t>(1),
              sum_spatial_multiplier_.mutable_cpu_data());
    buffer_spatial_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
  }
}

void NormalizeLayer::Forward_cpu(const vector<Blob*>& bottom,
                                 const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->cpu_data();
  real_t* top_data = top[0]->mutable_cpu_data();
  const real_t* scale = this->blobs_[0]->cpu_data();
  real_t* buffer_data = buffer_.mutable_cpu_data();
  real_t* norm_data = norm_.mutable_cpu_data();
  // add eps to avoid overflow
  caffe_set(norm_.count(), real_t(eps_), norm_data);
  const real_t* sum_channel_multiplier = sum_channel_multiplier_.cpu_data();
  const real_t* sum_spatial_multiplier = sum_spatial_multiplier_.cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / num;
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  int channels = bottom[0]->channels();
  for (int n = 0; n < num; ++n) {
    caffe_sqr(dim, bottom_data, buffer_data);
    if (across_spatial_) {
      // add eps to avoid overflow
      norm_data[n] = pow(caffe_cpu_asum(dim, buffer_data)+eps_,
                         real_t(0.5));
      caffe_cpu_scale(dim, real_t(1.0 / norm_data[n]), bottom_data,
                             top_data);
    } else {
      caffe_cpu_gemv(CblasTrans, channels, spatial_dim, real_t(1),
                            buffer_data, sum_channel_multiplier, real_t(1),
                            norm_data);
      // compute norm
      caffe_powx(spatial_dim, norm_data, real_t(0.5), norm_data);
      // scale the layer
      caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
                            1, real_t(1), sum_channel_multiplier, norm_data,
                            real_t(0), buffer_data);
      caffe_div(dim, bottom_data, buffer_data, top_data);
      norm_data += spatial_dim;
    }
    // scale the output
    if (channel_shared_) {
      caffe_scal(dim, scale[0], top_data);
    } else {
      caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
                            1, real_t(1), scale, sum_spatial_multiplier,
                            real_t(0),
                            buffer_data);
      caffe_mul(dim, top_data, buffer_data, top_data);
    }
    bottom_data += dim;
    top_data += dim;
  }
}

#ifndef USE_CUDA
STUB_GPU(NormalizeLayer);
#endif

REGISTER_LAYER_CLASS(Normalize);

}  // namespace caffe
