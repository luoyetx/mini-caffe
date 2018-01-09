#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>
#include <map>
#include "./common.hpp"
#include "./thread_local.hpp"

namespace caffe {

/*!
 * \brief Thread local MemoryPool
 *  This memory pool holds all memory for blobs in every thread.
 */
class MemoryPool {
 public:
  // small object size
  enum {
    kElementSize = 128,
    kPageSize = 1 << 20,  // 1 MB
  };

  using GpuKey = std::pair<int, size_t>;
  using CpuKey = size_t;
  struct MemBlock {
    int device{-1};
    size_t size{0};
    void* ptr{nullptr};
  };

  static MemoryPool* Get();
  /*!
   * \brief request memory from cpu
   * \param size memory size
   * \return memory block holds data size >= size
   */
  MemBlock RequestCPU(size_t size);
  /*!
   * \brief request memory from gpu
   * \param size memory size
   * \param device gpu device id
   * \return memory block holds data size >= size
   */
  MemBlock RequestGPU(size_t size, int device);
  /*! \brief return cpu memory block */
  void ReturnCPU(MemBlock cpu_block);
  /*! \brief return gpu memory block */
  void ReturnGPU(MemBlock gpu_block);
  /*! \brief get memory pool statistics */
  MemPoolState GetState();
  /*! \brief free all unused memory in pool */
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
