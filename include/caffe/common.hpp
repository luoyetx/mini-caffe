#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

#include <climits>
#include <cmath>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>  // pair
#include <vector>
#include <memory>

#include "caffe/logging.hpp"

#ifdef USE_CUDA

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types

//
// CUDA macros
//

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
    } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << caffe::cublasGetErrorString(status); \
    } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

namespace caffe {

  // CUDA: library error reporting.
  const char* cublasGetErrorString(cublasStatus_t error);
  const char* curandGetErrorString(curandStatus_t error);

  // CUDA: use 512 threads per block
  const int CAFFE_CUDA_NUM_THREADS = 512;

  // CUDA: number of blocks for threads.
  inline int CAFFE_GET_BLOCKS(const int N) {
    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
  }

}  // namespace caffe

#endif  // USE_CUDA

// Convert macro to string
#define STRINGIFY(m) #m
#define AS_STRING(m) STRINGIFY(m)

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname)            \
private:                                              \
  classname(const classname&) = delete;               \
  classname(classname&&) = delete;                    \
  classname& operator=(const classname&) = delete;    \
  classname& operator=(classname&&) = delete

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"
#define NO_GPU LOG(FATAL) << "Cannot use GPU in CPU-only Caffe: check mode."

#define STUB_GPU(classname)                                        \
void classname::Forward_gpu(const vector<Blob*>& bottom,           \
                            const vector<Blob*>& top) { NO_GPU; }                          

#define STUB_GPU_FORWARD(classname, funcname)                          \
void classname::funcname##_##gpu(const vector<Blob*>& bottom,          \
                                 const vector<Blob*>& top) { NO_GPU; }

#ifdef _MSC_VER
#ifdef CAFFE_EXPORTS
#define CAFFE_API __declspec(dllexport)
#else
#define CAFFE_API __declspec(dllimport)
#endif
#else
#define CAFFE_API
#endif

#ifdef _MSC_VER
#pragma warning(disable:4251)
#endif

namespace caffe {

// Common functions and classes from std that caffe often uses.
using std::shared_ptr;
using std::fstream;
using std::ios;
using std::isnan;
using std::isinf;
using std::iterator;
using std::make_pair;
using std::map;
using std::ostringstream;
using std::pair;
using std::set;
using std::string;
using std::stringstream;
using std::vector;

typedef float real_t;

// A singleton class to hold common caffe stuff, such as the handler that
// caffe is going to use for cublas, curand, etc.
class CAFFE_API Caffe {
public:
  ~Caffe();

  static Caffe& Get();

  enum Brew { CPU, GPU };

#ifdef USE_CUDA
  inline static cublasHandle_t cublas_handle() { return Get().cublas_handle_; }
#endif  // USE_CUDA

  // Returns the mode: running on CPU or GPU.
  inline static Brew mode() { return Get().mode_; }
  // The setters for the variables
  // Sets the mode. It is recommended that you don't change the mode halfway
  // into the program since that may cause allocation of pinned memory being
  // freed in a non-pinned way, which may cause problems - I haven't verified
  // it personally but better to note it here in the header file.
  inline static void set_mode(Brew mode) { Get().mode_ = mode; }
  // Sets the device. Since we have cublas and curand stuff, set device also
  // requires us to reset those values.
  static void SetDevice(const int device_id);
  // Check if specified device is available
  static bool CheckDevice(const int device_id);
  // Search from start_id to the highest possible device ordinal,
  // return the ordinal of the first available device.
  static int FindDevice(const int start_id = 0);

protected:
#ifdef USE_CUDA
  cublasHandle_t cublas_handle_;
#endif
  Brew mode_;

private:
  // The private constructor to avoid duplicate instantiation.
  Caffe();

  DISABLE_COPY_AND_ASSIGN(Caffe);
};

}  // namespace caffe

#endif  // CAFFE_COMMON_HPP_
