#include <misaka/op.h>

namespace misaka {
namespace op {

template<typename DType>
class ReLU : public Op {
  virtual void InferShape(const TensorShape& in,
                          TensorShape& tmp,
                          TensorShape& out) {
    tmp.clear();
    out.clear();
    CHECK_EQ(in.size(), 1);
    out.push_back(in[0]);
  }

  virtual void Call(const Tensor& in,
                    Tensor& tmp,
                    Tensor& out) {

  }
};

}
}