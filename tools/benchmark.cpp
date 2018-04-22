#include <stdlib.h>
#include <string>

#include <caffe/net.hpp>
#include <caffe/profiler.hpp>

int main(int argc, char *argv[]) {
  CHECK_EQ(argc, 4) << "[Usage]: ./benchmark net.prototxt iterations gpu_id";
  std::string proto = argv[1];
  int iters = atoi(argv[2]);
  int gpu_id = atoi(argv[3]);
  LOG(INFO) << "net prototxt: " << proto;
  LOG(INFO) << "net forward iterations: " << iters;
  LOG(INFO) << "run on device " << gpu_id;

  if (gpu_id >= 0 && caffe::GPUAvailable()) {
    caffe::SetMode(caffe::GPU, gpu_id);
  }
  else {
    gpu_id = -1;
  }

  caffe::Net net(proto);
  caffe::Profiler* profiler = caffe::Profiler::Get();
  profiler->TurnON();
  for (int i = 0; i < iters; i++) {
    uint64_t tic = profiler->Now();
    net.Forward();
    uint64_t toc = profiler->Now();
    LOG(INFO) << "Forward costs " << (toc - tic) / 1000. << " ms";
  }
  profiler->TurnOFF();
  profiler->DumpProfile("./profile.json");
  return 0;
}
