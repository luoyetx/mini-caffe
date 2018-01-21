#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <caffe/c_api.h>

#define CHECK(condition)                          \
  if (!(condition)) {                             \
    printf("CHECK (" #condition ") failed\n");    \
    exit(-1);                                     \
  }

#define CHECK_SUCCESS(condition)                  \
  if ((condition) != 0) {                         \
    printf("CHECK (" #condition ") failed\n");    \
    printf("%s\n", CaffeGetLastError());          \
    exit(-1);                                     \
  }

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
  int count = CaffeBlobCount(blob);
  CHECK(num == 1);
  CHECK(channels == 3);
  CHECK(height == 224);
  CHECK(width == 224);
  CHECK(count == num*channels*height*width);
  int shape[] = { num, channels, height, width };
  CHECK(CaffeBlobReshape(blob, 4, shape) == 0);
  // copy data
  real_t *data = CaffeBlobData(blob);
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

  // create network from buffer
  FILE *fin = fopen("model/resnet.prototxt", "r");
  fseek(fin, 0, SEEK_END);
  long prototxt_size = ftell(fin);
  fseek(fin, 0, SEEK_SET);
  char *prototxt = malloc(prototxt_size);
  fread(prototxt, 1, prototxt_size, fin);
  fclose(fin);
  fin = fopen("model/resnet.caffemodel", "rb");
  fseek(fin, 0, SEEK_END);
  long caffemodel_size = ftell(fin);
  fseek(fin, 0, SEEK_SET);
  char *caffemodel = malloc(caffemodel_size);
  fread(caffemodel, 1, caffemodel_size, fin);
  fclose(fin);
  CHECK_SUCCESS(CaffeNetCreateFromBuffer(prototxt, prototxt_size,
                                         caffemodel, caffemodel_size,
                                         &net));
  CHECK_SUCCESS(CaffeNetDestroy(net));
  return 0;
}
