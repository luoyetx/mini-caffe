#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>

#include "./crop_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void CropLayer::LayerSetUp(const vector<Blob*>& bottom,
                           const vector<Blob*>& top) {
  // LayerSetup() handles the number of dimensions; Reshape() handles the sizes.
  // bottom[0] supplies the data
  // bottom[1] supplies the size
  const CropParameter& param = this->layer_param_.crop_param();
  CHECK_EQ(bottom.size(), 2) << "Wrong number of bottom blobs.";
  int input_dim = bottom[0]->num_axes();
  const int start_axis = bottom[0]->CanonicalAxisIndex(param.axis());
  CHECK_LT(start_axis, input_dim) << "crop axis bigger than input dim";
  if (param.offset_size() > 1) {
    // the number of crop values specified must be equal to the number
    // of dimensions following axis
    CHECK_EQ(start_axis + param.offset_size(), input_dim)
      << "number of offset values specified must be equal to the number of "
      << "dimensions following axis.";
  }
}

void CropLayer::Reshape(const vector<Blob*>& bottom,
                        const vector<Blob*>& top) {
  const CropParameter& param = this->layer_param_.crop_param();
  int input_dim = bottom[0]->num_axes();
  const int start_axis = bottom[0]->CanonicalAxisIndex(param.axis());

  // Initialize offsets to 0 and the new shape to the current shape of the data.
  offsets = vector<int>(input_dim, 0);
  vector<int> new_shape(bottom[0]->shape());

  // Determine crop offsets and the new shape post-crop.
  for (int i = 0; i < input_dim; ++i) {
    int crop_offset = 0;
    int new_size = bottom[0]->shape(i);
    if (i >= start_axis) {
      new_size = bottom[1]->shape(i);
      if (param.offset_size() == 1) {
        // If only one offset is given, all crops have the same offset.
        crop_offset = param.offset(0);
      } else if (param.offset_size() > 1) {
        // For several offsets, the number of offsets must be equal to the
        // number of dimensions to crop, that is dimensions after the axis.
        crop_offset = param.offset(i - start_axis);
      }
      // Check that the crop and offset are within the dimension's bounds.
      CHECK_GE(bottom[0]->shape(i) - crop_offset, bottom[1]->shape(i))
          << "the crop for dimension " << i << " is out-of-bounds with "
          << "size " << bottom[1]->shape(i) << " and offset " << crop_offset;
    }
    new_shape[i] = new_size;
    offsets[i] = crop_offset;
  }
  top[0]->Reshape(new_shape);
}

void CropLayer::crop_copy(const vector<Blob*>& bottom,
                          const vector<Blob*>& top,
                          const vector<int>& offsets,
                          vector<int> indices,
                          int cur_dim,
                          const real_t* src_data,
                          real_t* dest_data,
                          bool is_forward) {
  if (cur_dim + 1 < top[0]->num_axes()) {
    // We are not yet at the final dimension, call copy recursively
    for (int i = 0; i < top[0]->shape(cur_dim); ++i) {
      indices[cur_dim] = i;
      crop_copy(bottom, top, offsets, indices, cur_dim+1,
                src_data, dest_data, is_forward);
    }
  } else {
    // We are at the last dimensions, which is stored continously in memory
    for (int i = 0; i < top[0]->shape(cur_dim); ++i) {
      // prepare index vector reduced(red) and with offsets(off)
      std::vector<int> ind_red(cur_dim, 0);
      std::vector<int> ind_off(cur_dim+1, 0);
      for (int j = 0; j < cur_dim; ++j) {
          ind_red[j] = indices[j];
          ind_off[j] = indices[j] + offsets[j];
      }
      ind_off[cur_dim] = offsets[cur_dim];
      // do the copy
      caffe_copy(top[0]->shape(cur_dim),
          src_data + bottom[0]->offset(ind_off),
          dest_data + top[0]->offset(ind_red));
    }
  }
}

void CropLayer::Forward_cpu(const vector<Blob*>& bottom,
                            const vector<Blob*>& top) {
  std::vector<int> indices(top[0]->num_axes(), 0);
  const real_t* bottom_data = bottom[0]->cpu_data();
  real_t* top_data = top[0]->mutable_cpu_data();
  crop_copy(bottom, top, offsets, indices, 0, bottom_data, top_data, true);
}

#ifndef USE_CUDA
STUB_GPU(CropLayer);
#endif

REGISTER_LAYER_CLASS(Crop);

}  // namespace caffe
