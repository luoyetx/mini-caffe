#ifndef CAFFE_CROP_LAYER_HPP_
#define CAFFE_CROP_LAYER_HPP_

#include <utility>
#include <vector>

#include "../layer.hpp"

namespace caffe {

/**
 * @brief Takes a Blob and crop it, to the shape specified by the second input
 *  Blob, across all dimensions after the specified axis.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */

class CropLayer : public Layer {
 public:
  explicit CropLayer(const LayerParameter& param)
      : Layer(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
                          const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
                       const vector<Blob*>& top);

  virtual const char* type() const { return "Crop"; }
  virtual int ExactNumBottomBlobs() const { return 2; }
  virtual int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);

  vector<int> offsets;

 private:
  // Recursive copy function.
  void crop_copy(const vector<Blob*>& bottom,
               const vector<Blob*>& top,
               const vector<int>& offsets,
               vector<int> indices,
               int cur_dim,
               const real_t* src_data,
               real_t* dest_data,
               bool is_forward);
  // Recursive copy function: this is similar to crop_copy() but loops over all
  // but the last two dimensions to allow for ND cropping while still relying on
  // a CUDA kernel for the innermost two dimensions for performance reasons.  An
  // alterantive implementation could rely on the kernel more by passing
  // offsets, but this is problematic because of its variable length.
  // Since in the standard (N,C,W,H) case N,C are usually not cropped a speedup
  // could be achieved by not looping the application of the copy_kernel around
  // these dimensions.
  void crop_copy_gpu(const vector<Blob*>& bottom,
                const vector<Blob*>& top,
                const vector<int>& offsets,
                vector<int> indices,
                int cur_dim,
                const real_t* src_data,
                real_t* dest_data,
                bool is_forward);
};

}  // namespace caffe

#endif  // CAFFE_CROP_LAYER_HPP_
