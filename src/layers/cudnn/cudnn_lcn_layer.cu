#ifdef USE_CUDNN

#include "./cudnn_lcn_layer.hpp"

namespace caffe {

void CuDNNLCNLayer::Forward_gpu(const vector<Blob*>& bottom,
                                const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();

  CUDNN_CHECK(cudnnDivisiveNormalizationForward(
              handle_, norm_desc_, CUDNN_DIVNORM_PRECOMPUTED_MEANS,
              cudnn::dataType<real_t>::one,
              bottom_desc_, bottom_data,
              NULL,  // srcMeansData
              this->tempData1, this->tempData2,
              cudnn::dataType<real_t>::zero,
              top_desc_, top_data));
}

}  // namespace caffe

#endif  // USE_CUDNN
