// Fillers are random number generators that fills a blob using the specified
// algorithm. The expectation is that they are only going to be used during
// initialization time and will not involve any GPUs.

#ifndef CAFFE_FILLER_HPP
#define CAFFE_FILLER_HPP

#include <string>

#include "caffe/blob.hpp"
#include "./util/math_functions.hpp"
#include "./proto/caffe.pb.h"

namespace caffe {

/// @brief Fills a Blob with constant or randomly-generated data.
class Filler {
 public:
  explicit Filler(const FillerParameter& param) : filler_param_(param) {}
  virtual ~Filler() {}
  virtual void Fill(Blob* blob) = 0;
 protected:
  FillerParameter filler_param_;
};  // class Filler


/// @brief Fills a Blob with constant values @f$ x = 0 @f$.
class ConstantFiller : public Filler {
 public:
  explicit ConstantFiller(const FillerParameter& param)
      : Filler(param) {}
  virtual void Fill(Blob* blob) {
    real_t* data = blob->mutable_cpu_data();
    const int count = blob->count();
    const real_t value = this->filler_param_.value();
    CHECK(count);
    for (int i = 0; i < count; ++i) {
      data[i] = value;
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/**
 * @brief Get a specific filler from the specification given in FillerParameter.
 *
 * Ideally this would be replaced by a factory pattern, but we will leave it
 * this way for now.
 */
inline Filler* GetFiller(const FillerParameter& param) {
  const std::string& type = param.type();
  if (type == "constant") {
    return new ConstantFiller(param);
  } else if (type == "gaussian") {
    return new ConstantFiller(param);
  } else if (type == "positive_unitball") {
    return new ConstantFiller(param);
  } else if (type == "uniform") {
    return new ConstantFiller(param);
  } else if (type == "xavier") {
    return new ConstantFiller(param);
  } else if (type == "msra") {
    return new ConstantFiller(param);
  } else if (type == "bilinear") {
    return new ConstantFiller(param);
  } else {
    CHECK(false) << "Unknown filler name: " << param.type();
  }
  return (Filler*)(NULL);
}

}  // namespace caffe

#endif  // CAFFE_FILLER_HPP_
