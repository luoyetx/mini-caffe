#ifndef CAFFE_UTIL_NNPACK_HPP_
#define CAFFE_UTIL_NNPACK_HPP_

#ifdef USE_NNPACK

#include <nnpack.h>

#include "caffe/common.hpp"

namespace caffe {

class NNPack {
public:
  static NNPack &Get() {
    static NNPack instance_;
    return instance_;
  }

  pthreadpool_t threadpool() {
    return threadpool_;
  }

private:
  NNPack() {
    nnp_status status = nnp_initialize();
    CHECK_EQ(status, nnp_status_success);
    const int num_threads = 1;  // TODO, need a better way
    threadpool_ = pthreadpool_create(num_threads);
  }
  ~NNPack() {
    nnp_status status = nnp_deinitialize();
    CHECK_EQ(status, nnp_status_success);
    pthreadpool_destroy(threadpool_);
  }

  pthreadpool_t threadpool_;

  DISABLE_COPY_AND_ASSIGN(NNPack);
};  // class NNPack

}  // namespace caffe
#endif  // USE_NNPACK

#endif  // CAFFE_UTIL_NNPACK_HPP_
