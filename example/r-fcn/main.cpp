#include <caffe/caffe.hpp>

int main(int argc, char* argv[]) {
  caffe::Net net("../models/r-fcn/faster-rcnn.prototxt");
  net.CopyTrainedLayersFrom("../models/r-fcn/faster-rcnn.caffemodel");
  return 0;
}
