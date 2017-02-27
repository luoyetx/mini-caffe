#!/usr/bin/env bash

set -e

./generatepb.sh

# build
mkdir build && cd build
cmake .. -DUSE_JAVA=ON -DBLAS=blas -DCMAKE_BUILD_TYPE=Release
make

# before test
URL_model="https://github.com/luoyetx/misc/blob/master/mini-caffe/model.zip?raw=true"
if [ ! -f model.zip ]; then
  echo "Downloading model.zip from $URL_model"
  wget -O model.zip $URL_model
fi
unzip -o model.zip

# test
./run_net
./run_net_c
cd ..

# java test
cd java
./gradlew clean build --info
cd ..

# python test
cd python
python2 --version
python2 tests/test.py
python2 setup.py build
python2 setup.py clean
python3 --version
python3 tests/test.py
python3 setup.py build
python3 setup.py clean
