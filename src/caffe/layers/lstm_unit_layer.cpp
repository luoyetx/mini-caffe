#include <algorithm>
#include <cmath>
#include <vector>

#include "./lstm_layer.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
inline Dtype tanh(Dtype x) {
  return 2. * sigmoid(2. * x) - 1.;
}

template <typename Dtype>
void LSTMUnitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int num_instances = bottom[0]->shape(1);
  for (int i = 0; i < bottom.size(); ++i) {
    if (i == 2) {
      CHECK_EQ(2, bottom[i]->num_axes());
    } else {
      CHECK_EQ(3, bottom[i]->num_axes());
    }
    CHECK_EQ(1, bottom[i]->shape(0));
    CHECK_EQ(num_instances, bottom[i]->shape(1));
  }
  hidden_dim_ = bottom[0]->shape(2);
  CHECK_EQ(num_instances, bottom[1]->shape(1));
  CHECK_EQ(4 * hidden_dim_, bottom[1]->shape(2));
  top[0]->ReshapeLike(*bottom[0]);
  top[1]->ReshapeLike(*bottom[0]);
  X_acts_.ReshapeLike(*bottom[1]);
}

template <typename Dtype>
void LSTMUnitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int num = bottom[0]->shape(1);
  const int x_dim = hidden_dim_ * 4;
  const Dtype* C_prev = bottom[0]->cpu_data();
  const Dtype* X = bottom[1]->cpu_data();
  const Dtype* cont = bottom[2]->cpu_data();
  Dtype* C = top[0]->mutable_cpu_data();
  Dtype* H = top[1]->mutable_cpu_data();
  for (int n = 0; n < num; ++n) {
    for (int d = 0; d < hidden_dim_; ++d) {
      const Dtype i = sigmoid(X[d]);
      const Dtype f = (*cont == 0) ? 0 :
          (*cont * sigmoid(X[1 * hidden_dim_ + d]));
      const Dtype o = sigmoid(X[2 * hidden_dim_ + d]);
      const Dtype g = tanh(X[3 * hidden_dim_ + d]);
      const Dtype c_prev = C_prev[d];
      const Dtype c = f * c_prev + i * g;
      C[d] = c;
      const Dtype tanh_c = tanh(c);
      H[d] = o * tanh_c;
    }
    C_prev += hidden_dim_;
    X += x_dim;
    C += hidden_dim_;
    H += hidden_dim_;
    ++cont;
  }
}

INSTANTIATE_CLASS(LSTMUnitLayer);
REGISTER_LAYER_CLASS(LSTMUnit);

}  // namespace caffe
