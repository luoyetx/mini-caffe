#ifdef USE_CUDNN

#include "./cudnn_lrn_layer.hpp"

namespace caffe {

void CuDNNLRNLayer::Forward_gpu(const vector<Blob*>& bottom,
                                const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();

  CUDNN_CHECK(cudnnLRNCrossChannelForward(
              handle_, norm_desc_, CUDNN_LRN_CROSS_CHANNEL_DIM1,
              cudnn::dataType<real_t>::one,
              bottom_desc_, bottom_data,
              cudnn::dataType<real_t>::zero,
              top_desc_, top_data));
}

};  // namespace caffe

#endif  // USE_CUDNN
