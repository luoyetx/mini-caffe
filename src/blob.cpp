#include <vector>

#include <google/protobuf/io/coded_stream.h>

#include "caffe/blob.hpp"
#include "./syncedmem.hpp"
#include "./util/io.hpp"
#include "./util/math_functions.hpp"
#include "./proto/caffe.pb.h"

namespace caffe {

void Blob::Reshape(const int num, const int channels,
                   const int height, const int width) {
  vector<int> shape(4);
  shape[0] = num;
  shape[1] = channels;
  shape[2] = height;
  shape[3] = width;
  Reshape(shape);
}

void Blob::Reshape(const vector<int>& shape) {
  CHECK_LE(shape.size(), kMaxBlobAxes);
  count_ = 1;
  shape_.resize(shape.size());
  if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)) {
    shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
  }
  int* shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());
  for (int i = 0; i < shape.size(); ++i) {
    CHECK_GE(shape[i], 0);
    if (count_ != 0) {
      CHECK_LE(shape[i], std::numeric_limits<int>::max() / count_) << "blob size exceeds INT_MAX";
    }
    count_ *= shape[i];
    shape_[i] = shape[i];
    shape_data[i] = shape[i];
  }
  if (count_ > capacity_) {
    capacity_ = count_;
    data_.reset(new SyncedMemory(capacity_ * sizeof(real_t)));
  }
}

void Blob::Reshape(const BlobShape& shape) {
  CHECK_LE(shape.dim_size(), kMaxBlobAxes);
  vector<int> shape_vec(shape.dim_size());
  for (int i = 0; i < shape.dim_size(); ++i) {
    shape_vec[i] = shape.dim(i);
  }
  Reshape(shape_vec);
}

void Blob::ReshapeLike(const Blob& other) {
  Reshape(other.shape());
}

Blob::Blob(const int num, const int channels,
           const int height, const int width)
    // capacity_ must be initialized before calling Reshape
    : capacity_(0) {
  Reshape(num, channels, height, width);
}

Blob::Blob(const vector<int>& shape)
    // capacity_ must be initialized before calling Reshape
    : capacity_(0) {
  Reshape(shape);
}

const int* Blob::gpu_shape() const {
  CHECK(shape_data_);
  return static_cast<const int*>(shape_data_->gpu_data());
}

const real_t* Blob::cpu_data() const {
  CHECK(data_);
  return static_cast<const real_t*>(data_->cpu_data());
}

void Blob::set_cpu_data(real_t* data) {
  CHECK(data);
  data_->set_cpu_data(data);
}

real_t* Blob::mutable_cpu_data() {
  CHECK(data_);
  return static_cast<real_t*>(data_->mutable_cpu_data());
}

const real_t* Blob::gpu_data() const {
  CHECK(data_);
  return static_cast<const real_t*>(data_->gpu_data());
}

real_t* Blob::mutable_gpu_data() {
  CHECK(data_);
  return static_cast<real_t*>(data_->mutable_gpu_data());
}

void Blob::ShareData(const Blob& other) {
  CHECK_EQ(count_, other.count());
  CHECK(other.data_);
  data_ = other.data_;
}

bool Blob::ShapeEquals(const BlobProto& other) {
  if (other.has_num() || other.has_channels() ||
      other.has_height() || other.has_width()) {
    // Using deprecated 4D Blob dimensions --
    // shape is (num, channels, height, width).
    // Note: we do not use the normal Blob::num(), Blob::channels(), etc.
    // methods as these index from the beginning of the blob shape, where legacy
    // parameter blobs were indexed from the end of the blob shape (e.g., bias
    // Blob shape (1 x 1 x 1 x N), IP layer weight Blob shape (1 x 1 x M x N)).
    return shape_.size() <= 4 &&
           LegacyShape(-4) == other.num() &&
           LegacyShape(-3) == other.channels() &&
           LegacyShape(-2) == other.height() &&
           LegacyShape(-1) == other.width();
  }
  vector<int> other_shape(other.shape().dim_size());
  for (int i = 0; i < other.shape().dim_size(); ++i) {
    other_shape[i] = other.shape().dim(i);
  }
  return shape_ == other_shape;
}

void Blob::CopyFrom(const Blob& source, bool reshape) {
  if (source.count() != count_ || source.shape() != shape_) {
    if (reshape) {
      ReshapeLike(source);
    } else {
      LOG(FATAL) << "Trying to copy blobs of different sizes.";
    }
  }
  caffe_copy(count_, source.cpu_data(),
             static_cast<real_t*>(data_->mutable_cpu_data()));
}

void Blob::FromProto(const BlobProto& proto, bool reshape) {
  if (reshape) {
    vector<int> shape;
    if (proto.has_num() || proto.has_channels() ||
        proto.has_height() || proto.has_width()) {
      // Using deprecated 4D Blob dimensions --
      // shape is (num, channels, height, width).
      shape.resize(4);
      shape[0] = proto.num();
      shape[1] = proto.channels();
      shape[2] = proto.height();
      shape[3] = proto.width();
    } else {
      shape.resize(proto.shape().dim_size());
      for (int i = 0; i < proto.shape().dim_size(); ++i) {
        shape[i] = proto.shape().dim(i);
      }
    }
    Reshape(shape);
  } else {
    CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
  }
  // copy data
  real_t* data_vec = mutable_cpu_data();
  if (proto.double_data_size() > 0) {
    CHECK_EQ(count_, proto.double_data_size());
    for (int i = 0; i < count_; ++i) {
      data_vec[i] = static_cast<real_t>(proto.double_data(i));
    }
  } else {
    CHECK_EQ(count_, proto.data_size());
    for (int i = 0; i < count_; ++i) {
      data_vec[i] = static_cast<real_t>(proto.data(i));
    }
  }
}

void Blob::ToProto(BlobProto* proto) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_data();
  proto->clear_diff();
  const float* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_data(data_vec[i]);
  }
}

const int* BlobInt::cpu_data() const {
  CHECK(data_);
  return static_cast<const int*>(data_->cpu_data());
}

void BlobInt::set_cpu_data(int* data) {
  CHECK(data);
  data_->set_cpu_data(data);
}

int* BlobInt::mutable_cpu_data() {
  CHECK(data_);
  return static_cast<int*>(data_->mutable_cpu_data());
}

const int* BlobInt::gpu_data() const {
  CHECK(data_);
  return static_cast<const int*>(data_->gpu_data());
}

int* BlobInt::mutable_gpu_data() {
  CHECK(data_);
  return static_cast<int*>(data_->mutable_gpu_data());
}

shared_ptr<Blob> ReadBlobFromFile(const string& file) {
  BlobProto bp;
  ReadProtoFromBinaryFileOrDie(file.c_str(), &bp);
  shared_ptr<Blob> blob(new Blob);
  blob->FromProto(bp);
  return blob;
}

shared_ptr<Blob> ReadBlobFromBuffer(const string& buffer) {
  using google::protobuf::uint8;
  google::protobuf::io::CodedInputStream ci(reinterpret_cast<const uint8*>(buffer.c_str()),
                                            buffer.length());
  BlobProto bp;
  CHECK(bp.ParseFromCodedStream(&ci)) << "Parse Blob failed";
  shared_ptr<Blob> blob;
  blob->FromProto(bp);
  return blob;
}

}  // namespace caffe
