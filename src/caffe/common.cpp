#include "caffe/common.hpp"

namespace caffe {

Caffe::Caffe()
  : mode_(Caffe::CPU) { }

Caffe::~Caffe() { }

Caffe& Caffe::Get() {
  static Caffe instance;
  return instance;
}

void Caffe::SetDevice(const int device_id) {
}

void Caffe::DeviceQuery() {
}

bool Caffe::CheckDevice(const int device_id) {
  return false;
}

int Caffe::FindDevice(const int start_id) {
  return -1;
}

}  // namespace caffe
