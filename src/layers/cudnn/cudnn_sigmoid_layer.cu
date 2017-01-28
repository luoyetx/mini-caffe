#ifdef USE_CUDNN

#include "./cudnn_sigmoid_layer.hpp"

namespace caffe {

void CuDNNSigmoidLayer::Forward_gpu(const vector<Blob*>& bottom,
                                    const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
#if CUDNN_VERSION_MIN(5, 0, 0)
  CUDNN_CHECK(cudnnActivationForward(this->handle_,
              activ_desc_,
              cudnn::dataType<real_t>::one,
              this->bottom_desc_, bottom_data,
              cudnn::dataType<real_t>::zero,
              this->top_desc_, top_data));
#else
  CUDNN_CHECK(cudnnActivationForward_v4(this->handle_,
              activ_desc_,
              cudnn::dataType<real_t>::one,
              this->bottom_desc_, bottom_data,
              cudnn::dataType<real_t>::zero,
              this->top_desc_, top_data));
#endif
}

}  // namespace caffe

#endif  // USE_CUDNN
