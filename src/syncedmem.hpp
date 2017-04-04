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
  struct GpuBlock {
    int device = -1;
    int size = 0;
    void* ptr = nullptr;

    GpuKey Key() const { return GpuKey{device, size}; }
  };
  using GpuBlockPair = std::pair<GpuKey, GpuBlock>;
  using CpuKey = int;
  struct CpuBlock {
    int size = 0;
    void* ptr = nullptr;

    CpuKey Key() const { return CpuKey{size}; }
  };
  using CpuBlockPair = std::pair<CpuKey, CpuBlock>;

  static MemoryPool* Get();

  CpuBlock RequestCPU(int size);
  GpuBlock RequestGPU(int size, int device);
  void ReturnCPU(CpuBlock cpu_block);
  void ReturnGPU(GpuBlock gpu_block);

 private:
  friend ThreadLocalStore<MemoryPool>;
  MemoryPool() {}
  ~MemoryPool();
  DISABLE_COPY_AND_ASSIGN(MemoryPool);

  std::multimap<CpuKey, CpuBlock> cpu_pool_;
  std::multimap<GpuKey, GpuBlock> gpu_pool_;
  std::multimap<CpuKey, CpuBlock> unused_cpu_pool_;
  std::multimap<GpuKey, GpuBlock> unused_gpu_pool_;
};

class SyncedMemory {
 public:
  //SyncedMemory()
  //    : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
  //    own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
  //    gpu_device_(-1) {}
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
  MemoryPool::CpuBlock cpu_block_;
  MemoryPool::GpuBlock gpu_block_;
  size_t size_;
  SyncedHead head_;

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
