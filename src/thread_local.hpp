// ThreadLocal Template
#ifndef CAFFE_THREAD_LOCAL_HPP_
#define CAFFE_THREAD_LOCAL_HPP_

#include <vector>
#include <mutex>

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

#endif  // CAFFE_THREAD_LOCAL_HPP_
