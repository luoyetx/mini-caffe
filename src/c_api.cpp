#include <mutex>

#include "caffe/c_api.h"
#include "caffe/blob.hpp"
#include "caffe/net.hpp"

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

int CaffeCreateNet(const char *net_path, const char *model_path,
                   NetHandle *net) {
  API_BEGIN();
  caffe::Net *net_ = new caffe::Net(net_path);
  net_->CopyTrainedLayersFrom(model_path);
  *net = static_cast<NetHandle>(net_);
  API_END();
}

int CaffeDestroyNet(NetHandle net) {
  API_BEGIN();
  delete static_cast<caffe::Net*>(net);
  API_END();
}

int CaffeForwardNet(NetHandle net) {
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

// Helper

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
