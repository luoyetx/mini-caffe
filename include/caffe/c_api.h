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

/*! \brief get blob num */
CAFFE_API int CaffeBlobNum(BlobHandle blob);
/*! \brief get blob channels */
CAFFE_API int CaffeBlobChannels(BlobHandle blob);
/*! \brief get blob height */
CAFFE_API int CaffeBlobHeight(BlobHandle blob);
/*! \brief get blob width */
CAFFE_API int CaffeBlobWidth(BlobHandle blob);
/*! \brief get blob data */
CAFFE_API real_t *CaffeBlobData(BlobHandle blob);
/*!
 * \brief reshape blob
 * \note  this may change blob data pointer
 */
CAFFE_API int CaffeBlobReshape(BlobHandle blob,
                               int num, int channels,
                               int height, int width);

// Net API
/*!
 * \brief create network
 * \param net_path path to network prototxt file
 * \param model_path path to network caffemodel file
 * \param net output NetHandle
 * \return return code, 0 for success, -1 for failed
 */
CAFFE_API int CaffeCreateNet(const char *net_path,
                             const char *model_path,
                             NetHandle *net);
/*! \brief destroy network */
CAFFE_API int CaffeDestroyNet(NetHandle net);
/*!
 * \brief forward network
 * \note  fill network input blobs before calling this function
 */
CAFFE_API int CaffeForwardNet(NetHandle net);
/*!
 * \brief get network internal blob by name
 * \param net NetHandle
 * \param name blob name
 * \param blob BlobHandle
 * \return return code
 */
CAFFE_API int CaffeNetGetBlob(NetHandle net,
                              const char *name,
                              BlobHandle *blob);
/*!
 * \brief return last API error info
 * \note  this function is thread safe
 */
CAFFE_API const char *CaffeGetLastError();

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // CAFFE_C_API_HPP_
