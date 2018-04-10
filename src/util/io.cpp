#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <stdint.h>

#include <algorithm>
#include <iostream>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>
#include <limits>

#include "./io.hpp"
#include "../proto/caffe.pb.h"

const int kProtoReadBytesLimit = std::numeric_limits<int>::max();  // Max size of 2 GB minus 1 byte.

namespace caffe {

using google::protobuf::io::IstreamInputStream;
using google::protobuf::io::OstreamOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::MessageLite;

bool ReadProtoFromTextFile(const char* filename, MessageLite* proto) {
  std::ifstream fin;
  fin.open(filename, std::ios::in);
  CHECK(fin.is_open()) << "File not found: " << filename;
  IstreamInputStream* input = new IstreamInputStream(&fin);
  bool success = proto->ParseFromZeroCopyStream(input);
  //bool success = google::protobuf::TextFormat::ParseFromString(input, proto);
  delete input;
  fin.close();
  return success;
}

void WriteProtoToTextFile(const MessageLite& proto, const char* filename) {
  std::ofstream fout;
  fout.open(filename, std::ios::out | std::ios::trunc);
  CHECK(fout.is_open()) << "Create file failed: " << filename;
  OstreamOutputStream* output = new OstreamOutputStream(&fout);
  CHECK(proto.SerializePartialToZeroCopyStream(output));
  //CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  fout.close();
}

bool ReadProtoFromBinaryFile(const char* filename, MessageLite* proto) {
  std::ifstream fin;
  fin.open(filename, std::ios::in | std::ios::binary);
  CHECK(fin.is_open()) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new IstreamInputStream(&fin);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  fin.close();
  return success;
}

void WriteProtoToBinaryFile(const MessageLite& proto, const char* filename) {
  std::ofstream fout;
  fout.open(filename, std::ios::out | std::ios::binary | std::ios::trunc);
  CHECK(fout.is_open()) << "Create file failed: " << filename;
  ZeroCopyOutputStream* raw_output = new OstreamOutputStream(&fout);
  CodedOutputStream* coded_output = new CodedOutputStream(raw_output);
  CHECK(proto.SerializeToCodedStream(coded_output));
  delete coded_output;
  delete raw_output;
  fout.close();
}

}  // namespace caffe
