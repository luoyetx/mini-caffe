#include <algorithm>
#include <vector>

#include "./shuffle_channel_layer.hpp"
#include "../util//math_functions.hpp"

namespace caffe {

void ShuffleChannelLayer::LayerSetUp(const vector<Blob*> &bottom,
                                     const vector<Blob*> &top) {
  group_ = this->layer_param_.shuffle_channel_param().group();
  CHECK_GT(group_, 0) << "group must be greater than 0";
  //temp_blob_.ReshapeLike(*bottom[0]);
  top[0]->ReshapeLike(*bottom[0]);
}

static void Resize_cpu(real_t* output, const real_t* input,
                       int group_row, int group_column, int len) {
  for (int i = 0; i < group_row; ++i) { // 2
    for(int j = 0; j < group_column ; ++j) { // 3
        const real_t* p_i = input + (i * group_column + j ) * len;
        real_t* p_o = output + (j * group_row + i ) * len;
        caffe_copy(len, p_i, p_o);
    }
  }
}

void ShuffleChannelLayer::Reshape(const vector<Blob*> &bottom,
                                  const vector<Blob*> &top) {
  int channels_ = bottom[0]->channels();
  int height_ = bottom[0]->height();
  int width_ = bottom[0]->width();
  top[0]->Reshape(bottom[0]->num(), channels_, height_, width_);
}

void ShuffleChannelLayer::Forward_cpu(const vector<Blob*>& bottom,
                                      const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->cpu_data();
  real_t* top_data = top[0]->mutable_cpu_data();

  const int num = bottom[0]->shape(0);
  const int feature_map_size = bottom[0]->count(1);
  const int sp_sz = bottom[0]->count(2);
  const int chs = bottom[0]->shape(1);

  int group_row = group_;
  int group_column = int(chs / group_row);
  CHECK_EQ(chs, (group_column * group_row)) << "Wrong group size.";

  //Dtype* temp_data = temp_blob_.mutable_cpu_data();
  for(int n = 0; n < num; ++n) {
	  Resize_cpu(top_data + n*feature_map_size, bottom_data + n*feature_map_size, group_row, group_column, sp_sz);
  }
  //caffe_copy(bottom[0]->count(), temp_blob_.cpu_data(), top_data);
}

#ifndef USE_CUDA
STUB_GPU(ShuffleChannelLayer);
#endif

REGISTER_LAYER_CLASS(ShuffleChannel);

}  // namespace caffe
