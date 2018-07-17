#include <vector>

#include "./mvn_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void MVNLayer::Reshape(const vector<Blob*>& bottom,
                       const vector<Blob*>& top) {
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  mean_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      1, 1);
  variance_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      1, 1);
  temp_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  if ( this->layer_param_.mvn_param().across_channels() ) {
    sum_multiplier_.Reshape(1, bottom[0]->channels(), bottom[0]->height(),
                            bottom[0]->width());
  } else {
    sum_multiplier_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
  }
  real_t* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), static_cast<real_t>(1), multiplier_data);
  eps_ = this->layer_param_.mvn_param().eps();
  // set temp blob name
  mean_.set_name(this->layer_param_.name() + "__mean__");
  variance_.set_name(this->layer_param_.name() + "__variance__");
  temp_.set_name(this->layer_param_.name() + "__temp__");
}

void MVNLayer::Forward_cpu(const vector<Blob*>& bottom,
                          const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->cpu_data();
  real_t* top_data = top[0]->mutable_cpu_data();
  int num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();

  int dim = bottom[0]->count() / num;

  // subtract mean
  caffe_cpu_gemv(CblasNoTrans, num, dim, 1. / dim, bottom_data,
    sum_multiplier_.cpu_data(), static_cast<real_t>(0), mean_.mutable_cpu_data());  // EX
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, static_cast<real_t>(-1),
    mean_.cpu_data(), sum_multiplier_.cpu_data(), static_cast<real_t>(0),
    temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

  if (this->layer_param_.mvn_param().normalize_variance()) {
    // compute variance using var(X) = E((X-EX)^2)
    caffe_powx(bottom[0]->count(), top_data, static_cast<real_t>(2),
      temp_.mutable_cpu_data());  // (X-EX)^2
    caffe_cpu_gemv(CblasNoTrans, num, dim, 1. / dim, temp_.cpu_data(),
      sum_multiplier_.cpu_data(), static_cast<real_t>(0),
      variance_.mutable_cpu_data());  // E((X-EX)^2)

    // normalize variance
    caffe_powx(variance_.count(), variance_.cpu_data(), static_cast<real_t>(0.5),
      variance_.mutable_cpu_data());

    caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());

    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, static_cast<real_t>(1),
      variance_.cpu_data(), sum_multiplier_.cpu_data(), static_cast<real_t>(0),
      temp_.mutable_cpu_data());

    caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
  }
}

#ifndef USE_CUDA
STUB_GPU(MVNLayer);
#endif

REGISTER_LAYER_CLASS(MVN);

}  // namespace caffe
