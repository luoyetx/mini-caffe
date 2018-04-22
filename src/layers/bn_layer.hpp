#ifndef CAFFE_BN_LAYER_HPP_
#define CAFFE_BN_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "../layer.hpp"

namespace caffe {

class BNLayer : public Layer {
 public:
	explicit BNLayer(const LayerParameter& param)
		: Layer(param) {}
	virtual void LayerSetUp(const vector<Blob*>& bottom,
		                      const vector<Blob*>& top);
	virtual void Reshape(const vector<Blob*>& bottom,
		                   const vector<Blob*>& top);
  virtual vector<Blob*> GetTempBlobs() { return{ &broadcast_buffer_, &spatial_statistic_, &x_norm_, &spatial_sum_multiplier_ }; }

	virtual const char* type() const { return "BN"; }
	virtual int ExactNumBottomBlobs() const { return 1; }
	virtual int ExactNumTopBlobs() const { return 1; }

 protected:
	virtual void Forward_cpu(const vector<Blob*>& bottom,
			                     const vector<Blob*>& top);
	virtual void Forward_gpu(const vector<Blob*>& bottom,
		                       const vector<Blob*>& top);

	void AverageAllExceptChannel(const real_t* input, real_t* output);
	void BroadcastChannel(const real_t* input, real_t* output);

	bool frozen_;
	real_t bn_momentum_;
	real_t bn_eps_;

	int num_;
	int channels_;
	int height_;
	int width_;

	Blob broadcast_buffer_;
	Blob spatial_statistic_;
	Blob batch_statistic_;

	Blob x_norm_;
	Blob x_inv_std_;

	Blob spatial_sum_multiplier_;
	Blob batch_sum_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_BN_LAYER_HPP_
