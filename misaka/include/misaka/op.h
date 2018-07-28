#pragma once

#include <vector>
#include <misaka/tensor.h>

namespace misaka {
namespace op {

using misaka::tensor::Tensor;
using misaka::tensor::TensorShape;

class Op {
 public:
  virtual ~Op();

  virtual void InferShape(std::vector<TensorShape>& in,
                          std::vector<TensorShape>& tmp,
                          std::vector<TensorShape>& out) = 0;
  virtual void Call(std::vector<Tensor>& in,
                    std::vector<Tensor>& tmp,
                    std::vector<Tensor>& out) = 0;
};

class OpManager {
 public:
  ~OpManager();

 private:

};

}
}