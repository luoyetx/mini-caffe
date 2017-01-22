#!/usr/bin/env bash

protoc -I="./src/proto" --cpp_out="./src/proto" "./src/proto/caffe.proto"

# build
mkdir build && cd build
cmake ..
make

# before test
URL_model="https://github.com/luoyetx/misc/blob/master/mini-caffe/model.zip?raw=true"
if [ ! -f model.zip ] then
  echo "Downloading model.zip from $URL_model"
  wget -O model.zip $URL_model
fi
unzip -o model.zip

# test
./run_net
