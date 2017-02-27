#include <mutex>

#include "caffe/c_api.h"
#include "caffe/blob.hpp"
#include "caffe/net.hpp"

// ThreadLocal Template

#ifdef __GNUC__
#define THREAD_LOCAL __thread
#elif __STDC_VERSION__ >= 201112L
#define THREAD_LOCAL _Thread_local
#elif defined(_MSC_VER)
#define THREAD_LOCAL __declspec(thread)
#endif

#ifndef THREAD_LOCAL
#message("Warning: Threadlocal is not enabled");
#endif

template<typename T>
class ThreadLocalStore {
public:
  static T *Get() {
    static THREAD_LOCAL T *ptr = nullptr;
    if (ptr == nullptr) {
      ptr = new T();
      Singleton()->RegisterDelete(ptr);
    }
    return ptr;
  }

private:
  ThreadLocalStore() {}
  ~ThreadLocalStore() {
    for (auto obj : objs_) {
      delete obj;
    }
  }
  static ThreadLocalStore<T> *Singleton() {
    static ThreadLocalStore<T> inst;
    return &inst;
  }
  void RegisterDelete(T *obj) {
    std::unique_lock<std::mutex> lock(mutex_);
    objs_.push_back(obj);
    lock.unlock();
  }

private:
  std::mutex mutex_;
  std::vector<T*> objs_;
};

#define API_BEGIN() try {
#define API_END() } catch(caffe::Error &_except_) { return CaffeAPIHandleException(_except_); } return 0;

void CaffeAPISetLastError(const char *msg);
int CaffeAPIHandleException(caffe::Error &e);

// API

int CaffeBlobNum(BlobHandle blob) {
  return static_cast<caffe::Blob*>(blob)->num();
}

int CaffeBlobChannels(BlobHandle blob) {
  return static_cast<caffe::Blob*>(blob)->channels();
}

int CaffeBlobHeight(BlobHandle blob) {
  return static_cast<caffe::Blob*>(blob)->height();
}

int CaffeBlobWidth(BlobHandle blob) {
  return static_cast<caffe::Blob*>(blob)->width();
}

real_t *CaffeBlobData(BlobHandle blob) {
  return static_cast<caffe::Blob*>(blob)->mutable_cpu_data();
}

int CaffeBlobReshape(BlobHandle blob, int num, int channels,
                     int height, int width) {
  API_BEGIN();
  static_cast<caffe::Blob*>(blob)->Reshape(num, channels, height, width);
  API_END();
}

int CaffeNetCreate(const char *net_path, const char *model_path,
                   NetHandle *net) {
  API_BEGIN();
  caffe::Net *net_ = new caffe::Net(net_path);
  net_->CopyTrainedLayersFrom(model_path);
  *net = static_cast<NetHandle>(net_);
  API_END();
}

int CaffeNetCreateFromBuffer(const char *net_buffer, int nb_len,
                             const char *model_buffer, int mb_len,
                             NetHandle *net) {
  API_BEGIN();
  std::shared_ptr<caffe::NetParameter> np;
  np = caffe::ReadBinaryNetParameterFromBuffer(model_buffer, mb_len);
  np = caffe::ReadTextNetParameterFromBuffer(net_buffer, nb_len);
  caffe::Net *net_ = new caffe::Net(*np.get());
  np = caffe::ReadBinaryNetParameterFromBuffer(model_buffer, mb_len);
  net_->CopyTrainedLayersFrom(*np.get());
  *net = static_cast<NetHandle>(net_);
  API_END();
}

int CaffeNetDestroy(NetHandle net) {
  API_BEGIN();
  delete static_cast<caffe::Net*>(net);
  API_END();
}

int CaffeNetForward(NetHandle net) {
  API_BEGIN();
  static_cast<caffe::Net*>(net)->Forward();
  API_END();
}

int CaffeNetGetBlob(NetHandle net, const char *name, BlobHandle *blob) {
  API_BEGIN();
  std::shared_ptr<caffe::Blob> blob_ = static_cast<caffe::Net*>(net)->blob_by_name(name);
  *blob = static_cast<BlobHandle>(blob_.get());
  API_END();
}

struct BlobsEntry {
  std::vector<const char*> vec_charp;
  std::vector<void*> vec_handle;
};

typedef ThreadLocalStore<BlobsEntry> BlobsStore;

int CaffeNetListBlob(NetHandle net, int *n, const char ***names, BlobHandle **blobs) {
  API_BEGIN();
  caffe::Net *net_ = static_cast<caffe::Net*>(net);
  const auto &names_ = net_->blob_names();
  const auto &blobs_ = net_->blobs();
  CHECK_EQ(names_.size(), blobs_.size());
  const int num = names_.size();
  auto *ret = BlobsStore::Get();
  ret->vec_charp.resize(num);
  ret->vec_handle.resize(num);
  for (int i = 0; i < num; i++) {
    ret->vec_charp[i] = names_[i].c_str();
    ret->vec_handle[i] = static_cast<BlobHandle>(blobs_[i].get());
  }
  *n = num;
  *names = ret->vec_charp.data();
  *blobs = ret->vec_handle.data();
  API_END();
}

int CaffeNetListParam(NetHandle net, int *n, const char ***names, BlobHandle **params) {
  API_BEGIN();
  caffe::Net *net_ = static_cast<caffe::Net*>(net);
  const auto &names_ = net_->param_names();
  const auto &params_ = net_->params();
  CHECK_EQ(names_.size(), params_.size());
  const int num = params_.size();
  auto *ret = BlobsStore::Get();
  ret->vec_charp.resize(num);
  ret->vec_handle.resize(num);
  for (int i = 0; i < num; i++) {
    ret->vec_charp[i] = names_[i].c_str();
    ret->vec_handle[i] = static_cast<BlobHandle>(params_[i].get());
  }
  *n = num;
  *names = ret->vec_charp.data();
  *params = ret->vec_handle.data();
  API_END();
}

int CaffeGPUAvailable() {
#ifdef USE_CUDA
  return 1;
#else
  return 0;
#endif  // USE_CUDA
}

int CaffeSetMode(int mode, int device) {
  API_BEGIN();
  if (mode == 0) {
    caffe::SetMode(caffe::CPU, -1);
  }
  else {
    CHECK_EQ(mode, 1);
    caffe::SetMode(caffe::GPU, device);
  }
  API_END();
}

// Helper

struct ErrorEntry {
  std::string last_error;
};

typedef ThreadLocalStore<ErrorEntry> ErrorStore;

void CaffeAPISetLastError(const char *msg) {
  ErrorStore::Get()->last_error = msg;
}

const char *CaffeGetLastError() {
  return ErrorStore::Get()->last_error.c_str();
}

int CaffeAPIHandleException(caffe::Error &e) {
  CaffeAPISetLastError(e.what());
  return -1;
}
