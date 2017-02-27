#include <algorithm>
#include <vector>

#include "./prelu_layer.hpp"
#include "../filler.hpp"

namespace caffe {

void PReLULayer::LayerSetUp(const vector<Blob*>& bottom,
                            const vector<Blob*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  PReLUParameter prelu_param = this->layer_param().prelu_param();
  int channels = bottom[0]->channels();
  channel_shared_ = prelu_param.channel_shared();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    if (channel_shared_) {
      this->blobs_[0].reset(new Blob(vector<int>(0)));
    } else {
      this->blobs_[0].reset(new Blob(vector<int>(1, channels)));
    }
    shared_ptr<Filler> filler;
    if (prelu_param.has_filler()) {
      filler.reset(GetFiller(prelu_param.filler()));
    } else {
      FillerParameter filler_param;
      filler_param.set_type("constant");
      filler_param.set_value(0.25);
      filler.reset(GetFiller(filler_param));
    }
    filler->Fill(this->blobs_[0].get());
  }
  if (channel_shared_) {
    CHECK_EQ(this->blobs_[0]->count(), 1)
        << "Negative slope size is inconsistent with prototxt config";
  } else {
    CHECK_EQ(this->blobs_[0]->count(), channels)
        << "Negative slope size is inconsistent with prototxt config";
  }
}

void PReLULayer::Reshape(const vector<Blob*>& bottom,
                         const vector<Blob*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  top[0]->ReshapeLike(*bottom[0]);
}

void PReLULayer::Forward_cpu(const vector<Blob*>& bottom,
                             const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->cpu_data();
  real_t* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const real_t* slope_data = this->blobs_[0]->cpu_data();

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  if (channel_shared_) {
    const float slop = slope_data[0];
    for (int i = 0; i < count; ++i) {
      top_data[i] = std::max(bottom_data[i], static_cast<real_t>(0))
          + slop * std::min(bottom_data[i], static_cast<real_t>(0));
    }
  }
  else {
    const int num = bottom[0]->num();
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < channels; j++) {
        const real_t slop = slope_data[j];
        for (int k = 0; k < dim; k++) {
          *top_data = std::max(*bottom_data, static_cast<real_t>(0))
              + slop * std::min(*bottom_data, static_cast<real_t>(0));
          top_data++;
          bottom_data++;
        }
      }
    }
  }
}

#ifndef USE_CUDA
STUB_GPU(PReLULayer);
#endif

REGISTER_LAYER_CLASS(PReLU);

}  // namespace caffe
