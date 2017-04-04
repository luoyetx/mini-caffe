#include "./common.hpp"
#include "./syncedmem.hpp"
#include "./util/math_functions.hpp"

namespace caffe {

using CpuBlock = MemoryPool::CpuBlock;
using GpuBlock = MemoryPool::GpuBlock;

static void CaffeMallocHost(CpuBlock& block, size_t size) {
  block = MemoryPool::Get()->RequestCPU(size);
}

static void CaffeFreeHost(CpuBlock block) {
  MemoryPool::Get()->ReturnCPU(block);
}

static void CaffeMallocDevice(GpuBlock& block, size_t size, int device) {
  block = MemoryPool::Get()->RequestGPU(size, device);
}

static void CaffeFreeDevice(GpuBlock block) {
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

MemoryPool::~MemoryPool() {
  for (auto it = cpu_pool_.begin(); it != cpu_pool_.end(); it++) {
    free(it->second.ptr);
  }
#ifdef USE_CUDA
  for (auto it = gpu_pool_.begin(); it != gpu_pool_.end(); it++) {
    cudaSetDevice(it->second.device);
    cudaError_t err = cudaFree(it->second.ptr);
    // ignore unloading error, as memory has already been recycled
    if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
      LOG(FATAL) << "CUDA: " << cudaGetErrorString(err);
    }
  }
#endif  // USE_CUDA
}

inline double MemSize(int size) {
  return std::round(static_cast<double>(size) / (1024 * 1024));
}

inline bool ShouldBorrowMem(int has, int wants) {
  static const int ratio = 2;
  return has / 2 < wants;
}

CpuBlock MemoryPool::RequestCPU(int size) {
  CpuKey key{size};
  auto it = unused_cpu_pool_.lower_bound(key);
  if (it == unused_cpu_pool_.end() || !ShouldBorrowMem(it->second.size,size)) {
    CpuBlock block;
    block.size = size;
    block.ptr = malloc(size);
    cpu_pool_.insert(CpuBlockPair{block.Key(), block});
    DLOG(INFO) << "[CPU] Requested " << MemSize(size) << ", Create " << MemSize(block.size);
    return block;
  }
  else {
    CpuBlock block = it->second;
    unused_cpu_pool_.erase(it);
    DLOG(INFO) << "[CPU] Requested " << MemSize(size) << ", Get " << MemSize(block.size);
    return block;
  }
}

void MemoryPool::ReturnCPU(CpuBlock block) {
  DLOG(INFO) << "[CPU] Return " << MemSize(block.size);
  unused_cpu_pool_.insert(CpuBlockPair{block.Key(), block});
}

GpuBlock MemoryPool::RequestGPU(int size, int device) {
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
    GpuBlock block;
    block.size = size;
    block.device = device;
    CUDA_CHECK(cudaMalloc(&block.ptr, size));
    if (cur_device != device) {
      CUDA_CHECK(cudaSetDevice(cur_device));
    }
    gpu_pool_.insert(GpuBlockPair{block.Key(), block});
    DLOG(INFO) << "[GPU] Requested " << MemSize(size) << ", Create " << MemSize(block.size);
    return block;
  }
  else {
    GpuBlock block = it->second;
    unused_gpu_pool_.erase(it);
    DLOG(INFO) << "[GPU] Requested " << MemSize(size) << ", Get " << MemSize(block.size);
    return block;
  }
#else
  NO_GPU;
  return GpuBlock();
#endif  // USE_CUDA
}

void MemoryPool::ReturnGPU(GpuBlock block) {
#ifdef USE_CUDA
  DLOG(INFO) << "[GPU] Return " << MemSize(block.size);
  unused_gpu_pool_.insert(GpuBlockPair{block.Key(), block});
#else
  NO_GPU;
#endif  // USE_CUDA
}

}  // namespace caffe
