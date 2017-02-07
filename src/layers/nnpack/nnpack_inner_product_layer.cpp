#ifdef USE_NNPACK

#include "./nnpack_inner_product_layer.hpp"
#include "../../util/math_functions.hpp"

namespace caffe {

void NNPackInnerProductLayer::Forward_cpu(const vector<Blob*>& bottom,
                                          const vector<Blob*>& top) {
  nnp_status status = nnp_status_success;
  if (M_ == 1) {
    status = nnp_fully_connected_inference(
        K_,
        N_,
        bottom[0]->cpu_data(),
        this->blobs_[0]->cpu_data(),
        top[0]->mutable_cpu_data(),
        NNPack::Get().threadpool());
  } else {
    status = nnp_fully_connected_output(
        M_,
        K_,
        N_,
        bottom[0]->cpu_data(),
        this->blobs_[0]->cpu_data(),
        top[0]->mutable_cpu_data(),
        NNPack::Get().threadpool(),
        nullptr);
  }
  CHECK_EQ(status, nnp_status_success);
  if (bias_term_) {
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, M_, N_, 1, static_cast<real_t>(1),
        bias_multiplier_.cpu_data(), this->blobs_[1]->cpu_data(),
        static_cast<real_t>(1), top[0]->mutable_cpu_data());
  }
}

}  // namespace caffe

#endif  // USE_NNPACK
