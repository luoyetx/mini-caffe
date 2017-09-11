#include <string>

#include <caffe/net.hpp>
#include <caffe/profiler.hpp>

int main(int argc, char *argv[]) {
  CHECK_EQ(argc, 4) << "[Usage]: ./benchmark net.prototxt net.caffemodel iterations";
  std::string proto = argv[1];
  std::string model = argv[2];
  int iters = std::stoi(argv[3]);
  LOG(INFO) << "net prototxt: " << proto;
  LOG(INFO) << "net caffemodel: " << model;
  LOG(INFO) << "net forward iterations: " << iters;

  caffe::Net net(proto);
  net.CopyTrainedLayersFrom(model);
  LOG(INFO) << "B";
  for (int i = 0; i < iters; i++) {
    net.Forward();
  }
  LOG(INFO) << "E";
  return 0;
}
