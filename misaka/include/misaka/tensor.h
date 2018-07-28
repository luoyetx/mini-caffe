#pragma once

#include <vector>

namespace misaka {
namespace tensor {

using TensorShape = std::vector<int32_t>;

enum class DeviceType {
  kCPU = 0,
  kGPU = 1,
};

enum class DataType {
  kFloat = 0,
  kDouble = 1,
};

class Tensor {
 public:
  Tensor(void* data, const TensorShape& shape, DeviceType dev_type, DataType data_type)
    : data_(data), shape(shape), dev_type(dev_type), data_type(data_type) {}

  TensorShape shape;
  DeviceType dev_type;
  DataType data_type;

  template<typename DType>
  const DType* data() const {
    return static_cast<DType>(this->data_);
  }

  template<typename DType>
  DType* data() {
    return static_cast<DType>(this->data);
  }

 private:
  void* data_;
};

}
}
