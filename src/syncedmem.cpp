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
    caffe_memset(size_, 0, cpu_block_.ptr);
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
    caffe_gpu_memset(size_, 0, gpu_block_.ptr);
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
  head_ = nullptr;
  curr_page_.device = -1;
  curr_page_.size = 0;
  curr_page_.ptr = nullptr;
  curr_ptr_ = kPageSize;  // used to trigger allocate
}

MemoryPool::~MemoryPool() {
  for (auto& block : cpu_pool_) {
    free(block.ptr);
  }
  cpu_pool_.clear();
#ifdef USE_CUDA
  for (auto& block : gpu_pool_) {
    cudaSetDevice(block.device);
    cudaError_t err = cudaFree(block.ptr);
    // ignore unloading error, as memory has already been recycled
    if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
      LOG(FATAL) << "CUDA: " << cudaGetErrorString(err);
    }
  }
  gpu_pool_.clear();
#endif  // USE_CUDA
}

inline double MemSize(int size) {
  return std::round(static_cast<double>(size) / (1024 * 1024) * 100) / 100;
}

inline bool ShouldBorrowMem(int has, int wants) {
  const int ratio = 2;
  return has / 2 <= wants;
}

MemBlock MemoryPool::RequestCPU(int size) {
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
        curr_page_.ptr = malloc(kPageSize);
        cpu_pool_.push_back(curr_page_);
        block.ptr = curr_page_.ptr;
        curr_ptr_ = kElementSize;
      }
    }
  }
  else {
    CpuKey key{size};
    auto it = unused_cpu_pool_.lower_bound(key);
    if (it == unused_cpu_pool_.end() || !ShouldBorrowMem(it->second.size, size)) {
      block.device = -1;
      block.size = size;
      block.ptr = malloc(size);
      cpu_pool_.push_back(block);
      DLOG(INFO) << "[CPU] Requested " << MemSize(size) << " M, Create " << MemSize(block.size) << " M";
    }
    else {
      block = it->second;
      unused_cpu_pool_.erase(it);
      DLOG(INFO) << "[CPU] Requested " << MemSize(size) << " M, Get " << MemSize(block.size) << " M";
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
    CpuKey key{ block.size };
    unused_cpu_pool_.insert(std::make_pair(key, block));
    DLOG(INFO) << "[CPU] Return " << MemSize(block.size) << " M";
  }
}

MemBlock MemoryPool::RequestGPU(int size, int device) {
  MemBlock block;
#ifdef USE_CUDA
  GpuKey key{device, size};
  auto it = unused_gpu_pool_.lower_bound(key);
  if (it == unused_gpu_pool_.end() || it->second.device != device ||
      !ShouldBorrowMem(it->second.size, size)) {
    int cur_device;
    CUDA_CHECK(cudaGetDevice(&cur_device));
    if (cur_device != device) {
      CUDA_CHECK(cudaSetDevice(device));
    }
    block.size = size;
    block.device = device;
    CUDA_CHECK(cudaMalloc(&block.ptr, size));
    if (cur_device != device) {
      CUDA_CHECK(cudaSetDevice(cur_device));
    }
    gpu_pool_.push_back(block);
    DLOG(INFO) << "[GPU] Requested " << MemSize(size) << " M, Create " << MemSize(block.size) << " M";
    return block;
  }
  else {
    block = it->second;
    unused_gpu_pool_.erase(it);
    DLOG(INFO) << "[GPU] Requested " << MemSize(size) << " M, Get " << MemSize(block.size) << " M";
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
  unused_gpu_pool_.insert(std::make_pair(key, block));
  DLOG(INFO) << "[GPU] Return " << MemSize(block.size) << " M";
#else
  NO_GPU;
#endif  // USE_CUDA
}

void MemoryPool::ClearUnused() {
  for (auto it = unused_cpu_pool_.begin(); it != unused_cpu_pool_.end(); it++) {
    free(it->second.ptr);
  }
  unused_cpu_pool_.clear();
#ifdef USE_CUDA
  for (auto it = unused_gpu_pool_.begin(); it != unused_gpu_pool_.end(); it++) {
    cudaSetDevice(it->second.device);
    cudaError_t err = cudaFree(it->second.ptr);
    // ignore unloading error, as memory has already been recycled
    if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
      LOG(FATAL) << "CUDA: " << cudaGetErrorString(err);
    }
  }
  unused_gpu_pool_.clear();
#endif  // USE_CUDA
}

MemPoolState MemoryPool::GetState() {
  MemPoolState st;
  st.cpu_mem = st.unused_cpu_mem = 0;
  st.gpu_mem = st.unused_gpu_mem = 0;
  for (auto& block : cpu_pool_) {
    st.cpu_mem += block.size;
  }
  for (auto it = unused_cpu_pool_.begin(); it != unused_cpu_pool_.end(); it++) {
    st.unused_cpu_mem += it->second.size;
  }
#ifdef USE_CUDA
  for (auto& block : gpu_pool_) {
    st.gpu_mem += block.size;
  }
  for (auto it = unused_gpu_pool_.begin(); it != unused_gpu_pool_.end(); it++) {
    st.unused_gpu_mem += it->second.size;
  }
#endif  // USE_CUDA
  return st;
}

void MemPoolClear() {
  MemoryPool::Get()->ClearUnused();
}

MemPoolState MemPoolGetState() {
  return MemoryPool::Get()->GetState();
}

}  // namespace caffe
