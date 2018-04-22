#ifndef CAFFE_PROPOSAL_LAYERS_HPP_
#define CAFFE_PROPOSAL_LAYERS_HPP_

#include <vector>

#include "../layer.hpp"

namespace caffe {

class ProposalLayer : public Layer {
 public:
  explicit ProposalLayer(const LayerParameter& param)
      : Layer(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
                          const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
                       const vector<Blob*>& top);

  virtual const char* type() const { return "ProposalLayer"; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);

  int base_size_;
  int feat_stride_;
  int pre_nms_topn_;
  int post_nms_topn_;
  real_t nms_thresh_;
  int min_size_;
  Blob anchors_;
  Blob proposals_;
  BlobInt roi_indices_;
  BlobInt nms_mask_;
};

}  // namespace caffe

#endif  // CAFFE_PROPOSAL_LAYERS_HPP_
