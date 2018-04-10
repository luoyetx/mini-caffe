#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "../filler.hpp"
#include "./normalize_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

// divid a matrix with vector
template <typename Dtype>
__global__ void DivBsx(const int nthreads, const Dtype* A,
    const Dtype* v, const int rows, const int cols, const CBLAS_TRANSPOSE trans,
    Dtype* B) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c = index % cols;
    int r = (index / cols) % rows;
    if (trans == CblasNoTrans) {
      B[index] = A[index] / v[c];
    } else {
      B[index] = A[index] / v[r];
    }
  }
}

template <typename Dtype>
__global__ void MulBsx(const int nthreads, const Dtype* A,
    const Dtype* v, const int rows, const int cols, const CBLAS_TRANSPOSE trans,
    Dtype* B) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c = index % cols;
    int r = (index / cols) % rows;
    if (trans == CblasNoTrans) {
      B[index] = A[index] * v[c];
    } else {
      B[index] = A[index] * v[r];
    }
  }
}

void NormalizeLayer::Forward_gpu(const vector<Blob*>& bottom,
                                 const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  real_t* buffer_data = buffer_.mutable_gpu_data();
  real_t* norm_data;
  if (across_spatial_) {
    // need to index it
    norm_data = norm_.mutable_cpu_data();
  } else {
    norm_data = norm_.mutable_gpu_data();
    // add eps to avoid overflow
    caffe_gpu_set(norm_.count(), real_t(eps_), norm_data);
  }
  const real_t* scale;
  if (channel_shared_) {
    scale = this->blobs_[0]->cpu_data();
  } else {
    scale = this->blobs_[0]->gpu_data();
  }
  const real_t* sum_channel_multiplier = sum_channel_multiplier_.gpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / num;
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  int channels = bottom[0]->channels();
  for (int n = 0; n < num; ++n) {
    caffe_gpu_powx(dim, bottom_data, real_t(2), buffer_data);
    if (across_spatial_) {
      real_t normsqr;
      caffe_gpu_asum(dim, buffer_data, &normsqr);
      // add eps to avoid overflow
      norm_data[n] = pow(normsqr+eps_, real_t(0.5));
      caffe_gpu_scale(dim, real_t(1.0 / norm_data[n]), bottom_data, top_data);
    } else {
      // compute norm
      caffe_gpu_gemv(CblasTrans, channels, spatial_dim, real_t(1),
                            buffer_data, sum_channel_multiplier, real_t(1),
                            norm_data);
      caffe_gpu_powx(spatial_dim, norm_data, real_t(0.5), norm_data);
      // scale the layer
      // NOLINT_NEXT_LINE(whitespace/operators)
      DivBsx<<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
          dim, bottom_data, norm_data, channels, spatial_dim, CblasNoTrans,
          top_data);
      CUDA_POST_KERNEL_CHECK;
      norm_data += spatial_dim;
    }
    // scale the output
    if (channel_shared_) {
      caffe_gpu_scal(dim, scale[0], top_data);
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      MulBsx<<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
          dim, top_data, scale, channels, spatial_dim, CblasTrans,
          top_data);
      CUDA_POST_KERNEL_CHECK;
    }
    bottom_data += dim;
    top_data += dim;
  }
}

}  // namespace caffe
