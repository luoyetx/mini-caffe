#ifndef CAFFE_C_API_HPP_
#define CAFFE_C_API_HPP_

#ifdef _MSC_VER
#ifdef CAFFE_EXPORTS
#define CAFFE_API __declspec(dllexport)
#else
#define CAFFE_API __declspec(dllimport)
#endif
#else
#define CAFFE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef float real_t;

typedef void *BlobHandle;
typedef void *NetHandle;

// Blob API
CAFFE_API int CaffeBlobNum(BlobHandle blob);
CAFFE_API int CaffeBlobChannels(BlobHandle blob);
CAFFE_API int CaffeBlobHeight(BlobHandle blob);
CAFFE_API int CaffeBlobWidth(BlobHandle blob);
CAFFE_API real_t *CaffeBlobData(BlobHandle blob);

// Net API
CAFFE_API NetHandle CaffeCreateNet(const char *net_path, const char *model_path);
CAFFE_API void CaffeDestroyNet(NetHandle net);
CAFFE_API void CaffeForwardNet(NetHandle net);
CAFFE_API BlobHandle CaffeNetGetBlob(NetHandle net, const char *name);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // CAFFE_C_API_HPP_
