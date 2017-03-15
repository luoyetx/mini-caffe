#ifdef USE_CUDNN

#include <vector>

#include "./cudnn_tanh_layer.hpp"

namespace caffe {

void CuDNNTanHLayer::Forward_gpu(const vector<Blob*>& bottom,
                                 const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  CUDNN_CHECK(cudnnActivationForward(this->handle_,
              activ_desc_,
              cudnn::dataType<real_t>::one,
              this->bottom_desc_, bottom_data,
              cudnn::dataType<real_t>::zero,
              this->top_desc_, top_data));
}

}  // namespace caffe

#endif  // USE_CUDNN
