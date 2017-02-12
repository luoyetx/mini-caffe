#ifdef USE_CUDNN

#include "./cudnn_bn_layer.hpp"

namespace caffe {

void CuDNNBNLayer::Forward_gpu(const vector<Blob*>& bottom,
                               const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  const real_t* scale_data = this->blobs_[0]->gpu_data();
  const real_t* bias_data = this->blobs_[1]->gpu_data();

  const real_t* running_mean_data = this->blobs_[2]->gpu_data();
  const real_t* running_inv_variance_data = this->blobs_[3]->gpu_data();
  CUDNN_CHECK(cudnnBatchNormalizationForwardInference(handle_,
      CUDNN_BATCHNORM_SPATIAL,
      cudnn::dataType<real_t>::one,
      cudnn::dataType<real_t>::zero,
      bottom_desc_, bottom_data,
      top_desc_, top_data,
      bn_param_desc_, scale_data, bias_data,
      running_mean_data, running_inv_variance_data,
      this->bn_eps_));
}

}  // namespace caffe

#endif  // USE_CUDNN
