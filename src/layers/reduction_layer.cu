#include <vector>

#include "./reduction_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void ReductionLayer::Forward_gpu(const vector<Blob*>& bottom,
                                 const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->gpu_data();
  const real_t* mult_data = NULL;
  if (sum_multiplier_.count() > 0) {
    mult_data = sum_multiplier_.gpu_data();
  }
  real_t* top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < num_; ++i) {
    switch (op_) {
    case ReductionParameter_ReductionOp_SUM:
    case ReductionParameter_ReductionOp_MEAN:
      caffe_gpu_dot(dim_, mult_data, bottom_data, top_data);
      break;
    case ReductionParameter_ReductionOp_ASUM:
      caffe_gpu_asum(dim_, bottom_data, top_data);
      break;
    case ReductionParameter_ReductionOp_SUMSQ:
      caffe_gpu_dot(dim_, bottom_data, bottom_data, top_data);
      break;
    default:
      LOG(FATAL) << "Unknown reduction op: "
          << ReductionParameter_ReductionOp_Name(op_);
    }
    bottom_data += dim_;
    ++top_data;
  }
  if (coeff_ != static_cast<real_t>(1)) {
    // Reset the top_data pointer.
    top_data = top[0]->mutable_gpu_data();
    caffe_gpu_scal(num_, coeff_, top_data);
  }
}

}  // namespace caffe
