#include <sstream>
#include <iomanip>
#include "./common.hpp"
#include "./syncedmem.hpp"
#include "./util/math_functions.hpp"

namespace caffe {

using MemBlock = MemoryPool::MemBlock;

static void CaffeMallocHost(MemBlock& block, size_t size) {
  block = MemoryPool::Get()->RequestCPU(size);
}

static void CaffeFreeHost(MemBlock block) {
  MemoryPool::Get()->ReturnCPU(block);
}

static void CaffeMallocDevice(MemBlock& block, size_t size, int device) {
  block = MemoryPool::Get()->RequestGPU(size, device);
}

static void CaffeFreeDevice(MemBlock block) {
  MemoryPool::Get()->ReturnGPU(block);
}

SyncedMemory::~SyncedMemory() {
  if (cpu_block_.ptr) {
    CaffeFreeHost(cpu_block_);
    cpu_block_.ptr = nullptr;
  }
#ifdef USE_CUDA
  if (gpu_block_.ptr) {
    CaffeFreeDevice(gpu_block_);
    gpu_block_.ptr = nullptr;
  }
#endif  // USE_CUDA
}

inline void SyncedMemory::to_cpu() {
  switch (head_) {
  case UNINITIALIZED:
    CaffeMallocHost(cpu_block_, size_);
    //caffe_memset(size_, 0, cpu_block_.ptr);
    head_ = HEAD_AT_CPU;
    break;
  case HEAD_AT_GPU:
#ifdef USE_CUDA
    if (cpu_block_.ptr == nullptr) {
      CaffeMallocHost(cpu_block_, size_);
    }
    caffe_gpu_memcpy(size_, gpu_block_.ptr, cpu_block_.ptr);
    head_ = SYNCED;
#else
    NO_GPU;
#endif  // USE_CUDA
    break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

inline void SyncedMemory::to_gpu() {
#ifdef USE_CUDA
  int device = -1;
  switch (head_) {
  case UNINITIALIZED:
    CUDA_CHECK(cudaGetDevice(&device));
    CaffeMallocDevice(gpu_block_, size_, device);
    //caffe_gpu_memset(size_, 0, gpu_block_.ptr);
    head_ = HEAD_AT_GPU;
    break;
  case HEAD_AT_CPU:
    if (gpu_block_.ptr == nullptr) {
      CUDA_CHECK(cudaGetDevice(&device));
      CaffeMallocDevice(gpu_block_, size_, device);
    }
    caffe_gpu_memcpy(size_, cpu_block_.ptr, gpu_block_.ptr);
    head_ = SYNCED;
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
#else
  NO_GPU;
#endif  // USE_CUDA
}

const void* SyncedMemory::cpu_data() {
  to_cpu();
  return (const void*)cpu_block_.ptr;
}

const void* SyncedMemory::gpu_data() {
#ifdef USE_CUDA
  to_gpu();
  return (const void*)gpu_block_.ptr;
#else
  NO_GPU;
  return nullptr;
#endif  // USE_CUDA
}

void* SyncedMemory::mutable_cpu_data() {
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_block_.ptr;
}

void* SyncedMemory::mutable_gpu_data() {
#ifdef USE_CUDA
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_block_.ptr;
#else
  NO_GPU;
  return nullptr;
#endif  // USE_CUDA
}

//// MemoryPool

MemoryPool* MemoryPool::Get() {
  return ThreadLocalStore<MemoryPool>::Get();
}

MemoryPool::MemoryPool() {
  // init small object pool
  head_ = nullptr;
  curr_page_.device = -1;
  curr_page_.size = 0;
  curr_page_.ptr = nullptr;
  curr_ptr_ = kPageSize;  // used to trigger allocate
  obj_pool_.clear();
  // init status
  st_.cpu_mem = st_.unused_cpu_mem = 0;
  st_.gpu_mem = st_.unused_gpu_mem = 0;
}

MemoryPool::~MemoryPool() {
  // all memory should be returned to pool
  Clear();
  // small object pool
  for (auto& block : obj_pool_) {
#ifdef USE_CUDA
    cudaError_t err = cudaFree(0);
    if (err != cudaErrorCudartUnloading) {
      CUDA_CHECK(cudaFreeHost(block.ptr));
    }
#else
    free(block.ptr);
#endif
  }
}

inline std::string MemSize(double size) {
  std::stringstream os;
  if (size < 1024.) {
    os << static_cast<int>(size) << " B";
  }
  else {
    size /= 1024.;
    os << std::setprecision(3);
    if (size < 1024.) {
      os << size << " K";
    }
    else {
      size /= 1024.;
      os << size << " M";
    }
  }
  return os.str();
}

inline bool ShouldBorrowMem(size_t has, size_t wants) {
  const int ratio = 2;
  return has / 2 <= wants;
}

MemBlock MemoryPool::RequestCPU(size_t size) {
  MemBlock block;
  if (size <= kElementSize) {  // small object <= 128 bytes
    block.device = -1;
    block.size = size;
    if (head_ != nullptr) {
      block.ptr = static_cast<void*>(head_);
      head_ = head_->next;
    }
    else {
      if (curr_ptr_ < kPageSize) {
        block.ptr = static_cast<void*>(static_cast<char*>(curr_page_.ptr) + curr_ptr_);
        curr_ptr_ += kElementSize;
      }
      else {
        curr_page_.device = -1;
        curr_page_.size = kPageSize;
#ifdef USE_CUDA
        CUDA_CHECK(cudaMallocHost(static_cast<void**>(&curr_page_.ptr), kPageSize));
#else
        curr_page_.ptr = malloc(kPageSize);
#endif
        st_.cpu_mem += kPageSize;
        obj_pool_.push_back(curr_page_);
        block.ptr = curr_page_.ptr;
        curr_ptr_ = kElementSize;
      }
    }
  }
  else {
    CpuKey key{size};
    auto it = cpu_pool_.lower_bound(key);
    if (it == cpu_pool_.end() || !ShouldBorrowMem(it->second.size, size)) {
      block.device = -1;
      block.size = size;
#ifdef USE_CUDA
      CUDA_CHECK(cudaMallocHost(static_cast<void**>(&block.ptr), size));
#else
      block.ptr = malloc(size);
#endif
      st_.cpu_mem += size;
      DLOG(INFO) << "[CPU] Requested " << MemSize(size) << ", Create " << MemSize(block.size);
    }
    else {
      block = it->second;
      cpu_pool_.erase(it);
      st_.unused_cpu_mem -= block.size;
      DLOG(INFO) << "[CPU] Requested " << MemSize(size) << ", Get " << MemSize(block.size);
    }
  }
  return block;
}

void MemoryPool::ReturnCPU(MemBlock block) {
  if (block.size <= kElementSize) {
    LinkedList* p = static_cast<LinkedList*>(block.ptr);
    p->next = head_;
    head_ = p;
  }
  else {
    CpuKey key{block.size};
    cpu_pool_.insert(std::make_pair(key, block));
    st_.unused_cpu_mem += block.size;
    DLOG(INFO) << "[CPU] Return " << MemSize(block.size);
  }
}

MemBlock MemoryPool::RequestGPU(size_t size, int device) {
  MemBlock block;
#ifdef USE_CUDA
  GpuKey key{device, size};
  auto it = gpu_pool_.lower_bound(key);
  if (it == gpu_pool_.end() || it->second.device != device ||
      !ShouldBorrowMem(it->second.size, size)) {
    int cur_device;
    CUDA_CHECK(cudaGetDevice(&cur_device));
    if (cur_device != device) {
      CUDA_CHECK(cudaSetDevice(device));
    }
    block.size = size;
    block.device = device;
    CUDA_CHECK(cudaMalloc(&block.ptr, size));
    st_.gpu_mem += size;
    if (cur_device != device) {
      CUDA_CHECK(cudaSetDevice(cur_device));
    }
    DLOG(INFO) << "[GPU] Requested " << MemSize(size) << ", Create " << MemSize(block.size);
    return block;
  }
  else {
    block = it->second;
    gpu_pool_.erase(it);
    st_.unused_gpu_mem -= block.size;
    DLOG(INFO) << "[GPU] Requested " << MemSize(size) << ", Get " << MemSize(block.size);
    return block;
  }
#else
  NO_GPU;
#endif  // USE_CUDA
  return block;
}

void MemoryPool::ReturnGPU(MemBlock block) {
#ifdef USE_CUDA
  GpuKey key{block.device, block.size};
  gpu_pool_.insert(std::make_pair(key, block));
  st_.unused_gpu_mem += block.size;
  DLOG(INFO) << "[GPU] Return " << MemSize(block.size);
#else
  NO_GPU;
#endif  // USE_CUDA
}

void MemoryPool::Clear() {
  for (auto it = cpu_pool_.begin(); it != cpu_pool_.end(); ++it) {
    MemBlock& block = it->second;
#ifdef USE_CUDA
    CUDA_CHECK(cudaFreeHost(block.ptr));
#else
    free(block.ptr);
#endif
    st_.cpu_mem -= block.size;
    st_.unused_cpu_mem -= block.size;
    DLOG(INFO) << "[CPU] Free " << MemSize(block.size);
  }
  cpu_pool_.clear();
#ifdef USE_CUDA
  int cur_device;
  cudaError_t err = cudaGetDevice(&cur_device);
  if (err == cudaErrorCudartUnloading) {
    // we are shutting down the program
    // ignore unloading error, as memory has already been recycled
    gpu_pool_.clear();
    return;
  }
  for (auto it = gpu_pool_.begin(); it != gpu_pool_.end(); ++it) {
    MemBlock& block = it->second;
    if (cur_device != block.device) {
      CUDA_CHECK(cudaSetDevice(block.device));
    }
    CUDA_CHECK(cudaFree(block.ptr));
    if (cur_device != block.device) {
      CUDA_CHECK(cudaSetDevice(cur_device));
    }
    st_.gpu_mem -= block.size;
    st_.unused_gpu_mem -= block.size;
    DLOG(INFO) << "[GPU] Free " << MemSize(block.size);
  }
  gpu_pool_.clear();
#endif  // USE_CUDA
}

MemPoolState MemoryPool::GetState() {
  int unused_cpu_mem = 0;
  int unused_gpu_mem = 0;
  for (auto it = cpu_pool_.begin(); it != cpu_pool_.end(); ++it) {
    unused_cpu_mem += it->second.size;
  }
  CHECK_EQ(unused_cpu_mem, st_.unused_cpu_mem);
#ifdef USE_CUDA
  for (auto it = gpu_pool_.begin(); it != gpu_pool_.end(); ++it) {
    unused_gpu_mem += it->second.size;
  }
  CHECK_EQ(unused_gpu_mem, st_.unused_gpu_mem);
#endif  // USE_CUDA
  return st_;
}

void MemPoolClear() {
  MemoryPool::Get()->Clear();
}

MemPoolState MemPoolGetState() {
  return MemoryPool::Get()->GetState();
}

}  // namespace caffe
