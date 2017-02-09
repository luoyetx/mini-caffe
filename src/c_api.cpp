#include "caffe/c_api.h"

#include "caffe/blob.hpp"
#include "caffe/net.hpp"

#define API_BEGIN()
#define API_END()

int CaffeBlobNum(BlobHandle blob) {
  API_BEGIN();
  return static_cast<caffe::Blob*>(blob)->num();
  API_END();
}

int CaffeBlobChannels(BlobHandle blob) {
  API_BEGIN();
  return static_cast<caffe::Blob*>(blob)->channels();
  API_END();
}

int CaffeBlobHeight(BlobHandle blob) {
  API_BEGIN();
  return static_cast<caffe::Blob*>(blob)->height();
  API_END();
}

int CaffeBlobWidth(BlobHandle blob) {
  API_BEGIN();
  return static_cast<caffe::Blob*>(blob)->width();
  API_END();
}

real_t *CaffeBlobData(BlobHandle blob) {
  API_BEGIN();
  return static_cast<caffe::Blob*>(blob)->mutable_cpu_data();
  API_END();
}

NetHandle CaffeCreateNet(const char *net_path, const char *model_path) {
  API_BEGIN();
  caffe::Net *net = new caffe::Net(net_path);
  net->CopyTrainedLayersFrom(model_path);
  return static_cast<NetHandle>(net);
  API_END();
}

void CaffeDestroyNet(NetHandle net) {
  API_BEGIN();
  delete static_cast<caffe::Net*>(net);
  API_END();
}

void CaffeForwardNet(NetHandle net) {
  API_BEGIN();
  static_cast<caffe::Net*>(net)->Forward();
  API_END();
}

BlobHandle CaffeNetGetBlob(NetHandle net, const char *name) {
  API_BEGIN();
  std::shared_ptr<caffe::Blob> blob = static_cast<caffe::Net*>(net)->blob_by_name(name);
  return static_cast<BlobHandle>(blob.get());
  API_END();
}
