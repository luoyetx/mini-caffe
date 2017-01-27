#ifndef CAFFE_TILE_LAYER_HPP_
#define CAFFE_TILE_LAYER_HPP_

#include <vector>

#include "../layer.hpp"

namespace caffe {

/**
 * @brief Copy a Blob along specified dimensions.
 */
class TileLayer : public Layer {
 public:
  explicit TileLayer(const LayerParameter& param)
      : Layer(param) {}
  virtual void Reshape(const vector<Blob*>& bottom,
                       const vector<Blob*>& top);

  virtual const char* type() const { return "Tile"; }
  virtual int ExactNumBottomBlobs() const { return 1; }
  virtual int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);

  unsigned int axis_, tiles_, outer_dim_, inner_dim_;
};

}  // namespace caffe

#endif  // CAFFE_TILE_LAYER_HPP_
