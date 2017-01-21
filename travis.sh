#!/usr/bin/env bash

protoc -I="./src/proto" --cpp_out="./src/proto" "./src/proto/caffe.proto"

mkdir build && cd build
cmake ..
make
