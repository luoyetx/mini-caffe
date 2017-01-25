#include <vector>

#include "./mvn_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void MVNLayer::Forward_gpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  int num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();

  int dim = bottom[0]->count() / num;

  // subtract mean
  caffe_gpu_gemv(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.gpu_data(), 0., mean_.mutable_gpu_data());  // EX
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
      temp_.mutable_gpu_data());
  caffe_gpu_add(temp_.count(), bottom_data, temp_.gpu_data(),
      top_data);  // X-EX

  if (this->layer_param_.mvn_param().normalize_variance()) {
    // compute variance using var(X) = E((X-EX)^2)
    caffe_gpu_powx(bottom[0]->count(), top_data, static_cast<real_t>(2),
        temp_.mutable_gpu_data());  // (X-EX)^2
    caffe_gpu_gemv(CblasNoTrans, num, dim, 1. / dim, temp_.gpu_data(),
        sum_multiplier_.gpu_data(), 0.,
        variance_.mutable_gpu_data());  // E((X-EX)^2)

    // normalize variance
    caffe_gpu_powx(variance_.count(), variance_.gpu_data(), static_cast<real_t>(0.5),
          variance_.mutable_gpu_data());

    caffe_gpu_add_scalar(variance_.count(), eps_, variance_.mutable_gpu_data());

    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
          temp_.mutable_gpu_data());

    caffe_gpu_div(temp_.count(), top_data, temp_.gpu_data(), top_data);
  }
}

}  // namespace caffe
