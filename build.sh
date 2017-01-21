#!/usr/bin/env bash

protoc -I="./src/caffe/proto" --cpp_out="./src/caffe/proto" "./src/caffe/proto/caffe.proto"

mkdir build && cd build
cmake ..
make
