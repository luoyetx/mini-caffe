#include <limits>
#include <random>

#include "./math_functions.hpp"

namespace caffe {

void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

void caffe_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

void caffe_axpy(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

void caffe_set(const int N, const real_t alpha, real_t* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(real_t) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

void caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

void caffe_copy(const int N, const real_t* X, real_t* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
#ifdef USE_CUDA
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(real_t) * N, cudaMemcpyDefault));
#else
      NO_GPU;
#endif
    }
    else {
      memcpy(Y, X, sizeof(real_t) * N);  // NOLINT(caffe/alt_fn)
    }
  }
}

void caffe_scal(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

void caffe_cpu_axpby(const int N, const float alpha, const float* X,
                     const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

void caffe_add(const int n, const float* a, const float* b, float* y) {
  vsAdd(n, a, b, y);
}

void caffe_sub(const int n, const float* a, const float* b, float* y) {
  vsSub(n, a, b, y);
}

void caffe_mul(const int n, const float* a, const float* b, float* y) {
  vsMul(n, a, b, y);
}

void caffe_div(const int n, const float* a, const float* b, float* y) {
  vsDiv(n, a, b, y);
}

void caffe_powx(const int n, const float* a, const float b, float* y) {
  vsPowx(n, a, b, y);
}

void caffe_sqr(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}

void caffe_exp(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

void caffe_log(const int n, const float* a, float* y) {
  vsLn(n, a, y);
}

void caffe_abs(const int n, const float* a, float* y) {
    vsAbs(n, a, y);
}

float caffe_cpu_strided_dot(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

real_t caffe_cpu_dot(const int n, const real_t* x, const real_t* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}

float caffe_cpu_asum(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

void caffe_cpu_scale(const int n, const float alpha, const float *x,
                     float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

}  // namespace caffe
