#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <caffe/c_api.h>

#define CHECK(condition)                          \
  if (!(condition)) {                             \
    printf("CHECK (" #condition ") failed\n");    \
    printf("%s\n", CaffeGetLastError());          \
    exit(-1);                                     \
  }

#define CHECK_SUCCESS(condition) CHECK((condition) == 0)

int main(int argc, char *argv[]) {
  // check gpu available
  if (CaffeGPUAvailable()) {
    CHECK_SUCCESS(CaffeSetMode(1, 0));
  }
  // create network
  NetHandle net;
  CHECK_SUCCESS(CaffeNetCreate("model/nin.prototxt", "model/nin.caffemodel", &net));
  // get data blob
  BlobHandle blob;
  CHECK_SUCCESS(CaffeNetGetBlob(net, "data", &blob));
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
  CHECK_SUCCESS(CaffeNetForward(net));
  clock_t end = clock();
  float time = (float)(end - start) / CLOCKS_PER_SEC;  // s
  time *= 1000;  // ms
  printf("Forward NIN costs %.4f ms\n", time);
  // list internal blobs
  int n;
  const char **names;
  BlobHandle *blobs;
  CHECK_SUCCESS(CaffeNetListBlob(net, &n, &names, &blobs));
  printf("NIN has %d internal data blobs\n", n);
  for (i = 0; i < n; i++) {
    printf("%s: [%d, %d, %d, %d]\n", names[i],
                                     CaffeBlobNum(blobs[i]),
                                     CaffeBlobChannels(blobs[i]),
                                     CaffeBlobHeight(blobs[i]),
                                     CaffeBlobWidth(blobs[i]));
  }
  // destroy
  CHECK_SUCCESS(CaffeNetDestroy(net));

  // should failed
  CHECK(CaffeNetCreate("no-such-prototxt", "no-such-caffemodel", &net) == -1);
  printf("%s\n", CaffeGetLastError());

  return 0;
}
