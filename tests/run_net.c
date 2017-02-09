#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <caffe/c_api.h>

#define CHECK(condition)                    \
  if (!(condition)) {                       \
    printf("CHECK " #condition " failed");  \
    exit(-1);                               \
  }

int main(int argc, char *argv[]) {
  // create network
  NetHandle net = CaffeCreateNet("model/nin.prototxt", "model/nin.caffemodel");
  // get data blob
  BlobHandle blob = CaffeNetGetBlob(net, "data");
  int num = CaffeBlobNum(blob);
  int channels = CaffeBlobChannels(blob);
  int height = CaffeBlobHeight(blob);
  int width = CaffeBlobWidth(blob);
  real_t *data = CaffeBlobData(blob);
  CHECK(num == 1);
  CHECK(channels == 3);
  CHECK(height == 224);
  CHECK(width == 224);
  // copy data
  int count = num * channels * height * width;
  int i;
  for (i = 0; i < count; i++) {
    float x = (float)(rand()) / RAND_MAX;  // 0 ~ 1
    data[i] = x * 256.f - 128.f;
  }
  // forward
  clock_t start = clock();
  CaffeForwardNet(net);
  clock_t end = clock();
  float time = (float)(end - start) / CLOCKS_PER_SEC;  // s
  time *= 1000;  // ms
  printf("Forward NIN costs %.4f ms\n", time);
  // destroy
  CaffeDestroyNet(net);
  return 0;
}
