#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>
#include <map>
#include "./common.hpp"
#include "./thread_local.hpp"

namespace caffe {

/*! \brief Thread local MemoryPool */
class MemoryPool {
 public:
  using GpuKey = std::pair<int, int>;
  using CpuKey = int;
  struct MemBlock {
    int device{-1};
    int size{0};
    void* ptr{nullptr};
  };

  static MemoryPool* Get();

  MemBlock RequestCPU(int size);
  MemBlock RequestGPU(int size, int device);
  void ReturnCPU(MemBlock cpu_block);
  void ReturnGPU(MemBlock gpu_block);

  MemPoolState GetState();
  void Clear();

 private:
  friend ThreadLocalStore<MemoryPool>;
  MemoryPool();
  ~MemoryPool();
  DISABLE_COPY_AND_ASSIGN(MemoryPool);

  //// pool for unused memory
  std::multimap<CpuKey, MemBlock> cpu_pool_;
  std::multimap<GpuKey, MemBlock> gpu_pool_;

  //// small object pool on CPU for size <= 128 bytes
  const int kMaxGPUs = 8;
  const int kElementSize = 128;
  const int kPageSize = 1 << 20;  // 1 MB
  struct LinkedList {
    LinkedList* next{nullptr};
  };
  LinkedList* head_;
  MemBlock curr_page_;
  int curr_ptr_;
  std::vector<MemBlock> obj_pool_;

  //// memory pool status
  MemPoolState st_;
};

class SyncedMemory {
 public:
  explicit SyncedMemory(size_t size)
      : cpu_block_(), gpu_block_(), size_(size), head_(UNINITIALIZED) {}
  ~SyncedMemory();
  const void* cpu_data();
  const void* gpu_data();
  void* mutable_cpu_data();
  void* mutable_gpu_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() { return head_; }
  size_t size() { return size_; }

 private:
  void to_cpu();
  void to_gpu();
  MemoryPool::MemBlock cpu_block_;
  MemoryPool::MemBlock gpu_block_;
  size_t size_;
  SyncedHead head_;

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
