#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#include <iomanip>
#include <iostream>  // NOLINT(readability/streams)
#include <string>

#include "google/protobuf/message_lite.h"

#include "caffe/base.hpp"
#include "../proto/caffe.pb.h"

#ifndef CAFFE_TMP_DIR_RETRIES
#define CAFFE_TMP_DIR_RETRIES 100
#endif

namespace caffe {

using ::google::protobuf::MessageLite;

bool ReadProtoFromTextFile(const char* filename, MessageLite* proto);

inline bool ReadProtoFromTextFile(const string& filename, MessageLite* proto) {
  return ReadProtoFromTextFile(filename.c_str(), proto);
}

inline void ReadProtoFromTextFileOrDie(const char* filename, MessageLite* proto) {
  CHECK(ReadProtoFromTextFile(filename, proto));
}

inline void ReadProtoFromTextFileOrDie(const string& filename, MessageLite* proto) {
  ReadProtoFromTextFileOrDie(filename.c_str(), proto);
}

void WriteProtoToTextFile(const MessageLite& proto, const char* filename);
inline void WriteProtoToTextFile(const MessageLite& proto, const string& filename) {
  WriteProtoToTextFile(proto, filename.c_str());
}

bool ReadProtoFromBinaryFile(const char* filename, MessageLite* proto);

inline bool ReadProtoFromBinaryFile(const string& filename, MessageLite* proto) {
  return ReadProtoFromBinaryFile(filename.c_str(), proto);
}

inline void ReadProtoFromBinaryFileOrDie(const char* filename, MessageLite* proto) {
  CHECK(ReadProtoFromBinaryFile(filename, proto));
}

inline void ReadProtoFromBinaryFileOrDie(const string& filename,
                                         MessageLite* proto) {
  ReadProtoFromBinaryFileOrDie(filename.c_str(), proto);
}

void WriteProtoToBinaryFile(const MessageLite& proto, const char* filename);
inline void WriteProtoToBinaryFile(
    const MessageLite& proto, const string& filename) {
  WriteProtoToBinaryFile(proto, filename.c_str());
}

}  // namespace caffe

#endif   // CAFFE_UTIL_IO_H_
