#include <vector>

#include "./embed_layer.hpp"
#include "../filler.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void EmbedLayer::LayerSetUp(const vector<Blob*>& bottom,
                            const vector<Blob*>& top) {
  N_ = this->layer_param_.embed_param().num_output();
  CHECK_GT(N_, 0) << "EmbedLayer num_output must be positive.";
  K_ = this->layer_param_.embed_param().input_dim();
  CHECK_GT(K_, 0) << "EmbedLayer input_dim must be positive.";
  bias_term_ = this->layer_param_.embed_param().bias_term();
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights --
    // transposed from InnerProductLayer for spatial locality.
    vector<int> weight_shape(2);
    weight_shape[0] = K_;
    weight_shape[1] = N_;
    this->blobs_[0].reset(new Blob(weight_shape));
    // fill the weights
    shared_ptr<Filler> weight_filler(GetFiller(
        this->layer_param_.embed_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob(bias_shape));
      shared_ptr<Filler> bias_filler(GetFiller(
          this->layer_param_.embed_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
}

void EmbedLayer::Reshape(const vector<Blob*>& bottom,
                         const vector<Blob*>& top) {
  // Figure out the dimensions
  M_ = bottom[0]->count();
  vector<int> top_shape = bottom[0]->shape();
  top_shape.push_back(N_);
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, static_cast<real_t>(1), bias_multiplier_.mutable_cpu_data());
  }
}

void EmbedLayer::Forward_cpu(const vector<Blob*>& bottom,
                             const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->cpu_data();
  const real_t* weight = this->blobs_[0]->cpu_data();
  real_t* top_data = top[0]->mutable_cpu_data();
  int index;
  for (int n = 0; n < M_; ++n) {
    index = static_cast<int>(bottom_data[n]);
    DCHECK_GE(index, 0);
    DCHECK_LT(index, K_);
    DCHECK_EQ(static_cast<real_t>(index), bottom_data[n]) << "non-integer input";
    caffe_copy(N_, weight + index * N_, top_data + n * N_);
  }
  if (bias_term_) {
    const real_t* bias = this->blobs_[1]->cpu_data();
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, M_, N_, 1, static_cast<real_t>(1),
      bias_multiplier_.cpu_data(), bias, static_cast<real_t>(1), top_data);
  }
}

#ifndef USE_CUDA
STUB_GPU(EmbedLayer);
#endif

REGISTER_LAYER_CLASS(Embed);

}  // namespace caffe
