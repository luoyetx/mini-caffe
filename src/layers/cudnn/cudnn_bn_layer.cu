#ifdef USE_CUDNN
#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/cudnn_bn_layer.hpp"

#if CUDNN_VERSION_MIN(4, 0, 0)

namespace caffe {

template <typename Dtype>
void CuDNNBNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* scale_data = this->blobs_[0]->gpu_data();
  const Dtype* bias_data = this->blobs_[1]->gpu_data();

  if (this->phase_ == TEST) {
    const Dtype* running_mean_data = this->blobs_[2]->gpu_data();
    const Dtype* running_inv_variance_data = this->blobs_[3]->gpu_data();
    CUDNN_CHECK(cudnnBatchNormalizationForwardInference(handle_,
        CUDNN_BATCHNORM_SPATIAL,
        cudnn::dataType<Dtype>::one,
        cudnn::dataType<Dtype>::zero,
        bottom_desc_, bottom_data,
        top_desc_, top_data,
        bn_param_desc_, scale_data, bias_data,
        running_mean_data, running_inv_variance_data,
        this->bn_eps_));
  } else {
    Dtype* running_mean_data = this->blobs_[2]->mutable_gpu_data();
    Dtype* running_inv_variance_data = this->blobs_[3]->mutable_gpu_data();
    Dtype* save_mean_data = save_mean_.mutable_gpu_data();
    Dtype* save_inv_variance_data = save_inv_variance_.mutable_gpu_data();
    CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(handle_,
        CUDNN_BATCHNORM_SPATIAL,
        cudnn::dataType<Dtype>::one,
        cudnn::dataType<Dtype>::zero,
        bottom_desc_, bottom_data,
        top_desc_, top_data,
        bn_param_desc_, scale_data, bias_data,
        this->bn_momentum_,
        running_mean_data, running_inv_variance_data,
        this->bn_eps_,
        save_mean_data, save_inv_variance_data));
  }
}

template <typename Dtype>
void CuDNNBNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0] || this->param_propagate_down_[0] ||
      this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* scale_data = this->blobs_[0]->gpu_data();
    Dtype* scale_diff = scale_buf_.mutable_gpu_diff();
    Dtype* bias_diff = bias_buf_.mutable_gpu_diff();
    const Dtype* save_mean_data = save_mean_.gpu_data();
    const Dtype* save_inv_variance_data = save_inv_variance_.gpu_data();

    CUDNN_CHECK(cudnnBatchNormalizationBackward(handle_,
        CUDNN_BATCHNORM_SPATIAL,
        cudnn::dataType<Dtype>::one,
        cudnn::dataType<Dtype>::zero,
        bottom_desc_, bottom_data,
        top_desc_, top_diff,
        bottom_desc_, bottom_diff,
        bn_param_desc_, scale_data,
        scale_diff, bias_diff,
        this->bn_eps_,
        save_mean_data, save_inv_variance_data));

    if (this->param_propagate_down_[0]) {
      caffe_gpu_add(scale_buf_.count(), scale_diff,
          this->blobs_[0]->gpu_diff(), this->blobs_[0]->mutable_gpu_diff());
    }
    if (this->param_propagate_down_[1]) {
      caffe_gpu_add(bias_buf_.count(), bias_diff,
          this->blobs_[1]->gpu_diff(), this->blobs_[1]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNBNLayer);

}  // namespace caffe
#endif
#endif
