#ifndef CAFFE_UTIL_MATH_FUNCTIONS_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <cstring>

#include "../common.hpp"
#include "./mkl_alternate.hpp"

namespace caffe {

// Caffe gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const real_t alpha, const real_t* A, const real_t* B, const real_t beta,
    real_t* C);

void caffe_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const real_t alpha, const real_t* A, const real_t* x, const real_t beta,
    real_t* y);

void caffe_axpy(const int N, const real_t alpha, const real_t* X,
    real_t* Y);

void caffe_cpu_axpby(const int N, const real_t alpha, const real_t* X,
    const real_t beta, real_t* Y);

void caffe_copy(const int N, const real_t *X, real_t *Y);

void caffe_set(const int N, const real_t alpha, real_t *X);

inline void caffe_memset(const size_t N, const int alpha, void* X) {
  memset(X, alpha, N);  // NOLINT(caffe/alt_fn)
}

void caffe_add_scalar(const int N, const real_t alpha, real_t *X);

void caffe_scal(const int N, const real_t alpha, real_t *X);

void caffe_sqr(const int N, const real_t* a, real_t* y);

void caffe_add(const int N, const real_t* a, const real_t* b, real_t* y);

void caffe_sub(const int N, const real_t* a, const real_t* b, real_t* y);

void caffe_mul(const int N, const real_t* a, const real_t* b, real_t* y);

void caffe_div(const int N, const real_t* a, const real_t* b, real_t* y);

void caffe_powx(const int n, const real_t* a, const real_t b, real_t* y);

void caffe_exp(const int n, const real_t* a, real_t* y);

void caffe_log(const int n, const real_t* a, real_t* y);

void caffe_abs(const int n, const real_t* a, real_t* y);

real_t caffe_cpu_dot(const int n, const real_t* x, const real_t* y);

real_t caffe_cpu_strided_dot(const int n, const real_t* x, const int incx,
    const real_t* y, const int incy);

// Returns the sum of the absolute values of the elements of vector x
real_t caffe_cpu_asum(const int n, const real_t* x);

void caffe_cpu_scale(const int n, const real_t alpha, const real_t *x, real_t* y);

#ifdef USE_CUDA  // GPU

// Decaf gpu gemm provides an interface that is almost the same as the cpu
// gemm function - following the c convention and calling the fortran-order
// gpu code under the hood.
void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const real_t alpha, const real_t* A, const real_t* B, const real_t beta,
    real_t* C);

void caffe_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const real_t alpha, const real_t* A, const real_t* x, const real_t beta,
    real_t* y);

void caffe_gpu_axpy(const int N, const real_t alpha, const real_t* X,
    real_t* Y);

void caffe_gpu_axpby(const int N, const real_t alpha, const real_t* X,
    const real_t beta, real_t* Y);

void caffe_gpu_memcpy(const size_t N, const void *X, void *Y);

void caffe_gpu_set(const int N, const real_t alpha, real_t *X);

inline void caffe_gpu_memset(const size_t N, const int alpha, void* X) {
  CUDA_CHECK(cudaMemset(X, alpha, N));  // NOLINT(caffe/alt_fn)
}

void caffe_gpu_add_scalar(const int N, const real_t alpha, real_t *X);

void caffe_gpu_scal(const int N, const real_t alpha, real_t *X);

void caffe_gpu_add(const int N, const real_t* a, const real_t* b, real_t* y);

void caffe_gpu_sub(const int N, const real_t* a, const real_t* b, real_t* y);

void caffe_gpu_mul(const int N, const real_t* a, const real_t* b, real_t* y);

void caffe_gpu_div(const int N, const real_t* a, const real_t* b, real_t* y);

void caffe_gpu_abs(const int n, const real_t* a, real_t* y);

void caffe_gpu_exp(const int n, const real_t* a, real_t* y);

void caffe_gpu_log(const int n, const real_t* a, real_t* y);

void caffe_gpu_powx(const int n, const real_t* a, const real_t b, real_t* y);

void caffe_gpu_dot(const int n, const real_t* x, const real_t* y, real_t* out);

void caffe_gpu_asum(const int n, const real_t* x, real_t* y);

void caffe_gpu_scale(const int n, const real_t alpha, const real_t *x, real_t* y);

#endif  // USE_CUDA

}  // namespace caffe

#endif  // CAFFE_UTIL_MATH_FUNCTIONS_H_
