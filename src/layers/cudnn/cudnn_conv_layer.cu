#ifdef USE_CUDNN

#include "./cudnn_conv_layer.hpp"

namespace caffe {

__global__ void sync_conv_groups() { }

void CuDNNConvolutionLayer::Forward_gpu(const vector<Blob*>& bottom,
                                        const vector<Blob*>& top) {
  const real_t* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const real_t* bottom_data = bottom[i]->gpu_data();
    real_t* top_data = top[i]->mutable_gpu_data();

    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      // Filters.
      CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
                  cudnn::dataType<real_t>::one,
                  bottom_descs_[i], bottom_data + bottom_offset_ * g,
                  filter_desc_, weight + this->weight_offset_ * g,
                  conv_descs_[i],
                  fwd_algo_[i], workspace[g], workspace_fwd_sizes_[i],
                  cudnn::dataType<real_t>::zero,
                  top_descs_[i], top_data + top_offset_ * g));

      // Bias.
      if (this->bias_term_) {
        const real_t* bias_data = this->blobs_[1]->gpu_data();
        CUDNN_CHECK(cudnnAddTensor(handle_[g],
                    cudnn::dataType<real_t>::one,
                    bias_desc_, bias_data + bias_offset_ * g,
                    cudnn::dataType<real_t>::one,
                    top_descs_[i], top_data + top_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups<<<1, 1>>>();
  }
}

}  // namespace caffe

#endif  // USE_CUDNN
