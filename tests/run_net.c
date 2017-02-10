#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <caffe/c_api.h>

#define CHECK(condition)                          \
  if (!(condition)) {                             \
    printf("CHECK (" #condition ") failed\n");    \
    exit(-1);                                     \
  }

int main(int argc, char *argv[]) {
  // create network
  NetHandle net;
  CHECK(CaffeNetCreate("model/nin.prototxt", "model/nin.caffemodel", &net) == 0);
  // get data blob
  BlobHandle blob;
  CHECK(CaffeNetGetBlob(net, "data", &blob) == 0);
  int num = CaffeBlobNum(blob);
  int channels = CaffeBlobChannels(blob);
  int height = CaffeBlobHeight(blob);
  int width = CaffeBlobWidth(blob);
  CHECK(num == 1);
  CHECK(channels == 3);
  CHECK(height == 224);
  CHECK(width == 224);
  CHECK(CaffeBlobReshape(blob, num, channels, height, width) == 0);
  // copy data
  real_t *data = CaffeBlobData(blob);
  int count = num * channels * height * width;
  int i;
  for (i = 0; i < count; i++) {
    float x = (float)(rand()) / RAND_MAX;  // 0 ~ 1
    data[i] = x * 256.f - 128.f;
  }
  // forward
  clock_t start = clock();
  CHECK(CaffeNetForward(net) == 0);
  clock_t end = clock();
  float time = (float)(end - start) / CLOCKS_PER_SEC;  // s
  time *= 1000;  // ms
  printf("Forward NIN costs %.4f ms\n", time);
  // list internal blobs
  int n;
  const char **names;
  BlobHandle *blobs;
  CHECK(CaffeNetListBlob(net, &n, &names, &blobs) == 0);
  printf("NIN has %d internal data blobs\n", n);
  for (i = 0; i < n; i++) {
    printf("%s: [%d, %d, %d, %d]\n", names[i],
                                     CaffeBlobNum(blobs[i]),
                                     CaffeBlobChannels(blobs[i]),
                                     CaffeBlobHeight(blobs[i]),
                                     CaffeBlobWidth(blobs[i]));
  }
  // destroy
  CHECK(CaffeNetDestroy(net) == 0);

  // should failed
  CHECK(CaffeNetCreate("no-such-prototxt", "no-such-caffemodel", &net) == -1);
  printf("%s\n", CaffeGetLastError());

  return 0;
}
