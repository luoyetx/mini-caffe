#!/usr/bin/env bash

protoc -I="./src/proto" --cpp_out="./src/proto" --python_out="./tools" "./src/proto/caffe.proto"
