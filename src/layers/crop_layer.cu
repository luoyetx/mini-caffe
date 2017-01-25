#include <vector>

#include "./crop_layer.hpp"

namespace caffe {

// Copy (one line per thread) from one array to another, with arbitrary
// strides in the last two dimensions.
__global__ void copy_kernel(const int n, const int height, const int width,
    const int src_outer_stride, const int src_inner_stride,
    const int dest_outer_stride, const int dest_inner_stride,
    const real_t* src, real_t* dest) {
  CUDA_KERNEL_LOOP(index, n) {
    int src_start = index / height * src_outer_stride
                  + index % height * src_inner_stride;
    int dest_start = index / height * dest_outer_stride
                   + index % height * dest_inner_stride;
    for (int i = 0; i < width; ++i) {
      dest[dest_start + i] = src[src_start + i];
    }
  }
}

void CropLayer::crop_copy_gpu(const vector<Blob*>& bottom,
                              const vector<Blob*>& top,
                              const vector<int>& offsets,
                              vector<int> indices,
                              int cur_dim,
                              const real_t* src_data,
                              real_t* dest_data,
                              bool is_forward) {
  if (cur_dim + 2 < top[0]->num_axes()) {
    // We are not yet at the final dimension, call copy recursivley
    for (int i = 0; i < top[0]->shape(cur_dim); ++i) {
      indices[cur_dim] = i;
      crop_copy_gpu(bottom, top, offsets, indices, cur_dim+1,
                src_data, dest_data, is_forward);
    }
  } else {
    // We are at the last two dimensions, which are stored continously in memory
    // With (N,C,H,W)
    //      (0,1,2,3) cur_dim   -> H
    //                cur_dim+1 -> W
    const int lines = top[0]->shape(cur_dim);
    const int height = top[0]->shape(cur_dim);
    const int width = top[0]->shape(cur_dim+1);
    std::vector<int> ind_off(cur_dim+2, 0);
    for (int j = 0; j < cur_dim; ++j) {
        ind_off[j] = indices[j] + offsets[j];
    }
    ind_off[cur_dim] = offsets[cur_dim];
    ind_off[cur_dim+1] = offsets[cur_dim+1];
    // Compute copy strides
    const int src_outer_stride =
        bottom[0]->shape(cur_dim)*bottom[0]->shape(cur_dim+1);
    const int src_inner_stride = bottom[0]->shape(cur_dim+1);
    const int dest_outer_stride =
        top[0]->shape(cur_dim)*top[0]->shape(cur_dim+1);
    const int dest_inner_stride = top[0]->shape(cur_dim+1);

    const real_t* bottom_data = bottom[0]->gpu_data() +
        bottom[0]->offset(ind_off);
    real_t* top_data = top[0]->mutable_gpu_data() +
        top[0]->offset(indices);
    // NOLINT_NEXT_LINE(whitespace/operators)
    copy_kernel<<<CAFFE_GET_BLOCKS(lines), CAFFE_CUDA_NUM_THREADS>>>(
        lines, height, width,
        src_outer_stride, src_inner_stride,
        dest_outer_stride, dest_inner_stride,
        bottom_data, top_data);
  }
}

void CropLayer::Forward_gpu(const vector<Blob*>& bottom,
                            const vector<Blob*>& top) {
  std::vector<int> indices(top[0]->num_axes(), 0);
  const real_t* bottom_data = bottom[0]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  crop_copy_gpu(bottom, top, offsets, indices, 0, bottom_data, top_data, true);
}

}  // namespace caffe
