#include <jni.h>
#include <string.h>
#include "caffe/c_api.h"

#define CaffeJNIMethodName(klass, method) Java_com_luoyetx_minicaffe_##klass##_jni##method
#define CaffeJNIMethod(klass, method, return_type) \
  JNIEXPORT return_type JNICALL CaffeJNIMethodName(klass, method)

#define CHECK_SUCCESS(condition, ...)   \
  if ((condition) != 0) {               \
    __VA_ARGS__                         \
    return -1;                          \
  } else {                              \
    __VA_ARGS__                         \
  }

#define JNIGetHandleFromObj(obj, handle) do {                       \
    jclass kls = (*env)->GetObjectClass(env, obj);                  \
    jfieldID field = (*env)->GetFieldID(env, kls, "handle", "J");   \
    handle = (void*)((*env)->GetLongField(env, obj, field));        \
  } while(0)

#define JNISetHandleToObj(obj, handle) do {                         \
    jclass kls = (*env)->GetObjectClass(env, obj);                  \
    jfieldID field = (*env)->GetFieldID(env, kls, "handle", "J");   \
    (*env)->SetLongField(env, obj, field, (jlong)handle);           \
  } while(0)

// class Blob

CaffeJNIMethod(Blob, SyncToJava, jint)(JNIEnv *env, jobject thiz) {
  BlobHandle blob;
  JNIGetHandleFromObj(thiz, blob);
  int shape_size = 0;
  int* shape_data = NULL;
  CaffeBlobShape(blob, &shape_size, &shape_data);
  float *data = CaffeBlobData(blob);
  // set meta data
  jclass kls = (*env)->GetObjectClass(env, thiz);
  jintArray java_shape = (*env)->NewIntArray(env, shape_size);
  (*env)->SetIntArrayRegion(env, java_shape, 0, shape_size, shape_data);
  jfieldID field = (*env)->GetFieldID(env, kls, "shape", "[I");
  (*env)->SetObjectField(env, thiz, field, java_shape);
  // set float array data
  int length = CaffeBlobCount(blob);
  jfloatArray java_data = (*env)->NewFloatArray(env, length);
  (*env)->SetFloatArrayRegion(env, java_data, 0, length, data);
  field = (*env)->GetFieldID(env, kls, "data", "[F");
  (*env)->SetObjectField(env, thiz, field, java_data);
  return 0;
}

CaffeJNIMethod(Blob, SyncToC, jint)(JNIEnv *env, jobject thiz) {
  BlobHandle blob;
  JNIGetHandleFromObj(thiz, blob);
  // get meta
  jclass kls = (*env)->GetObjectClass(env, thiz);
  jfieldID field = (*env)->GetFieldID(env, kls, "shape", "[I");
  jintArray shape = (*env)->GetObjectField(env, thiz, field);
  jint* shape_data = (*env)->GetPrimitiveArrayCritical(env, shape, NULL);
  int shape_size = (*env)->GetArrayLength(env, shape);
  int length = CaffeBlobCount(blob);
  CHECK_SUCCESS(CaffeBlobReshape(blob, shape_size, shape_data), {
    (*env)->ReleasePrimitiveArrayCritical(env, shape, shape_data, 0);
  });
  // get float array data
  field = (*env)->GetFieldID(env, kls, "data", "[F");
  jfloatArray java_data = (*env)->GetObjectField(env, thiz, field);
  float *data = CaffeBlobData(blob);
  jfloat *data_ = (*env)->GetPrimitiveArrayCritical(env, java_data, NULL);
  memcpy(data, data_, length * sizeof(float));
  (*env)->ReleasePrimitiveArrayCritical(env, java_data, data_, 0);
  return 0;
}

// class Net

CaffeJNIMethod(Net, Create, jint)(JNIEnv *env, jobject thiz,
                                  jstring net_path, jstring model_path) {
  NetHandle net;
  const char *net_path_cstr = (*env)->GetStringUTFChars(env, net_path, NULL);
  const char *model_path_cstr = (*env)->GetStringUTFChars(env, model_path, NULL);
  CHECK_SUCCESS(CaffeNetCreate(net_path_cstr, model_path_cstr, &net), {
    (*env)->ReleaseStringUTFChars(env, net_path, net_path_cstr);
    (*env)->ReleaseStringUTFChars(env, model_path, model_path_cstr);
  });
  JNISetHandleToObj(thiz, net);
  return 0;
}

CaffeJNIMethod(Net, CreateFromBuffer, jint)(JNIEnv *env, jobject thiz,
                                            jbyteArray net_buffer,
                                            jbyteArray model_buffer) {
  NetHandle net;
  jbyte *nbuffer = (*env)->GetByteArrayElements(env, net_buffer, NULL);
  jbyte *mbuffer = (*env)->GetByteArrayElements(env, model_buffer, NULL);
  jsize nb_len = (*env)->GetArrayLength(env, net_buffer);
  jsize mb_len = (*env)->GetArrayLength(env, model_buffer);
  CHECK_SUCCESS(CaffeNetCreateFromBuffer((const char*)nbuffer, nb_len,
                                         (const char*)mbuffer, mb_len, &net), {
    (*env)->ReleaseByteArrayElements(env, net_buffer, nbuffer, 0);
    (*env)->ReleaseByteArrayElements(env, model_buffer, mbuffer, 0);
  });
  JNISetHandleToObj(thiz, net);
  return 0;
}

CaffeJNIMethod(Net, Destroy, jint)(JNIEnv *env, jobject thiz) {
  NetHandle net;
  JNIGetHandleFromObj(thiz, net);
  CHECK_SUCCESS(CaffeNetDestroy(net));
  JNISetHandleToObj(thiz, NULL);
  return 0;
}

CaffeJNIMethod(Net, MarkOutput, jint)(JNIEnv *env, jobject thiz,
                                      jstring name) {
  NetHandle net;
  JNIGetHandleFromObj(thiz, net);
  const char *name_cstr = (*env)->GetStringUTFChars(env, name, NULL);
  CHECK_SUCCESS(CaffeNetMarkOutput(net, name_cstr), {
    (*env)->ReleaseStringUTFChars(env, name, name_cstr);
  });
  return 0;
}

CaffeJNIMethod(Net, Forward, jint)(JNIEnv *env, jobject thiz) {
  NetHandle net;
  JNIGetHandleFromObj(thiz, net);
  CHECK_SUCCESS(CaffeNetForward(net));
  return 0;
}

CaffeJNIMethod(Net, GetBlob, jint)(JNIEnv *env, jobject thiz,
                                   jstring name, jobject blob) {
  NetHandle net;
  BlobHandle blob_;
  JNIGetHandleFromObj(thiz, net);
  const char *name_cstr = (*env)->GetStringUTFChars(env, name, NULL);
  CHECK_SUCCESS(CaffeNetGetBlob(net, name_cstr, &blob_), {
    (*env)->ReleaseStringUTFChars(env, name, name_cstr);
  });
  // set blob handle
  JNISetHandleToObj(blob, blob_);
  CaffeJNIMethodName(Blob, SyncToJava)(env, blob);
  return 0;
}

// class Utils

CaffeJNIMethod(Utils, GetLastError, jstring)(JNIEnv *env, jobject thiz) {
  const char *msg = CaffeGetLastError();
  return (*env)->NewStringUTF(env, msg);
}

CaffeJNIMethod(Utils, GPUAvailable, jint)(JNIEnv *env, jobject thiz) {
  return CaffeGPUAvailable();
}

CaffeJNIMethod(Utils, SetMode, jint)(JNIEnv *env, jobject thiz,
                                     jint mode, jint device) {
  return CaffeSetMode(mode, device);
}

CaffeJNIMethod(Utils, ProfilerEnable, jint)(JNIEnv *env, jobject thiz) {
  return CaffeProfilerEnable();
}

CaffeJNIMethod(Utils, ProfilerDisable, jint)(JNIEnv *env, jobject thiz) {
  return CaffeProfilerDisable();
}

CaffeJNIMethod(Utils, ProfilerScopeStart, jint)(JNIEnv *env, jobject thiz,
                                                jstring name) {
  const char *name_cstr = (*env)->GetStringUTFChars(env, name, NULL);
  CHECK_SUCCESS(CaffeProfilerScopeStart(name_cstr), {
    (*env)->ReleaseStringUTFChars(env, name, name_cstr);
  });
  return 0;
}

CaffeJNIMethod(Utils, ProfilerScopeEnd, jint)(JNIEnv *env, jobject thiz) {
  return CaffeProfilerScopeEnd();
}

CaffeJNIMethod(Utils, ProfilerDump, jint)(JNIEnv *env, jobject thiz,
                                          jstring fn) {
  const char *fn_cstr = (*env)->GetStringUTFChars(env, fn, NULL);
  CHECK_SUCCESS(CaffeProfilerDump(fn_cstr), {
    (*env)->ReleaseStringUTFChars(env, fn, fn_cstr);
  });
  return 0;
}
