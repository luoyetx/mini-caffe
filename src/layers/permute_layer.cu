#include <algorithm>
#include <cfloat>
#include <vector>

#include "./permute_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

__global__ void PermuteKernel(const int nthreads,
    real_t* const bottom_data, const bool forward, const int* permute_order,
    const int* old_steps, const int* new_steps, const int num_axes,
    real_t* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int temp_idx = index;
    int old_idx = 0;
    for (int i = 0; i < num_axes; ++i) {
      int order = permute_order[i];
      old_idx += (temp_idx / new_steps[i]) * old_steps[order];
      temp_idx %= new_steps[i];
    }
    if (forward) {
      top_data[index] = bottom_data[old_idx];
    } else {
      bottom_data[old_idx] = top_data[index];
    }
  }
}

void PermuteLayer::Forward_gpu(const vector<Blob*>& bottom,
                               const vector<Blob*>& top) {
  if (need_permute_) {
    real_t* bottom_data = bottom[0]->mutable_gpu_data();
    real_t* top_data = top[0]->mutable_gpu_data();
    int count = top[0]->count();
    const int* permute_order = permute_order_.gpu_data();
    const int* new_steps = new_steps_.gpu_data();
    const int* old_steps = old_steps_.gpu_data();
    bool foward = true;
    // NOLINT_NEXT_LINE(whitespace/operators)
    PermuteKernel<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, foward, permute_order, old_steps, new_steps,
        num_axes_, top_data);
    CUDA_POST_KERNEL_CHECK;
  } else {
    // If there is no need to permute
    caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
  }
}

}  // namespace caffe
