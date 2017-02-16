#!/usr/bin/env bash

set -e

# set up envs
ANDROID_ROOT=`pwd`
THIRD_PARTY_ROOT=$ANDROID_ROOT/../3rdparty/src
ANDROID_TOOLCHAIN_FILE=$ANDROID_ROOT/android-cmake/android.toolchain.cmake

# manually setting
ANDROID_ABI=arm64-v8a
ANDROID_NATIVE_API_LEVEL=21
ANDROID_BUILD_JOBS=2

echo "Android Build Root: $ANDROID_ROOT"
echo "Android NDK Root: $NDK_ROOT"
echo "Android ABI: $ANDROID_ABI"
echo "Android Native API Level: $ANDROID_NATIVE_API_LEVEL"

# update submodule
echo "Update git submodules"
git submodule update --init

# build protobuf
echo "Build protobuf"
cd $ANDROID_ROOT
PROTOBUF_INSTALL_ROOT=$ANDROID_ROOT/install
mkdir -p $PROTOBUF_INSTALL_ROOT
mkdir -p protobuf-build
cd protobuf-build
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_TOOLCHAIN_FILE \
      -DANDROID_NDK=$NDK_ROOT \
      -DANDROID_ABI=$ANDROID_ABI \
      -DANDROID_NATIVE_API_LEVEL=$ANDROID_NATIVE_API_LEVEL \
      -DCMAKE_INSTALL_PREFIX=$PROTOBUF_INSTALL_ROOT \
      -DCMAKE_BUILD_TYPE=Release \
      -Dprotobuf_BUILD_TESTS=OFF \
      -Dprotobuf_BUILD_SHARED_LIBS=OFF \
      $THIRD_PARTY_ROOT/protobuf/cmake
make -j$ANDROID_BUILD_JOBS
make install

# build protobuf for host
echo "Build protobuf for host"
cd $ANDROID_ROOT
PROTOBUF_HOST_INSTALL_ROOT=$ANDROID_ROOT/protobuf-host-build
mkdir -p $PROTOBUF_HOST_INSTALL_ROOT
mkdir -p protobuf-host-build
cd protobuf-host-build
cmake -DCMAKE_INSTALL_PREFIX=$PROTOBUF_HOST_INSTALL_ROOT \
      -DCMAKE_BUILD_TYPE=Release \
      -Dprotobuf_BUILD_TESTS=OFF \
      -Dprotobuf_BUILD_SHARED_LIBS=OFF \
      $THIRD_PARTY_ROOT/protobuf/cmake
make -j$ANDROID_BUILD_JOBS
make install

# build OpenBLAS
echo "Build OpenBLAS"
cd $ANDROID_ROOT
OPENBLAS_INSTALL_ROOT=$ANDROID_ROOT/install
cd $THIRD_PARTY_ROOT/OpenBLAS

CROSS_SUFFIX=$NDK_ROOT/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/bin/aarch64-linux-android-
SYSROOT=$NDK_ROOT/platforms/android-$ANDROID_NATIVE_API_LEVEL/arch-arm64
TARGET=ARMV8
BINARY=64

# make -j$ANDROID_BUILD_JOBS \
#      CC="${CROSS_SUFFIX}gcc --sysroot=$SYSROOT" \
#      FC="${CROSS_SUFFIX}gfortran --sysroot=$SYSROOT" \
#      CROSS_SUFFIX=$CROSS_SUFFIX \
#      HOSTCC=gcc USE_THREAD=1 NUM_THREADS=4 USE_OPENMP=1 \
#      NO_LAPACK=1 TARGET=$TARGET BINARY=$BINARY
make -j$ANDROID_BUILD_JOBS \
     CC="${CROSS_SUFFIX}gcc --sysroot=$SYSROOT" \
     NOFORTRAN=1 \
     CROSS_SUFFIX=$CROSS_SUFFIX \
     HOSTCC=gcc USE_THREAD=1 NUM_THREADS=4 \
     NO_LAPACK=1 TARGET=$TARGET BINARY=$BINARY
make PREFIX=$OPENBLAS_INSTALL_ROOT install

# build MiniCaffe
echo "Build MiniCaffe"
cd $ANDROID_ROOT
mkdir -p MiniCaffe-build
cd MiniCaffe-build
MINI_CAFFE_ROOT=$ANDROID_ROOT/../
echo "protoc $MINI_CAFFE_ROOT/src/proto/caffe.proto"
$PROTOBUF_HOST_INSTALL_ROOT/bin/protoc \
    -I="$MINI_CAFFE_ROOT/src/proto" \
    --cpp_out="$MINI_CAFFE_ROOT/src/proto" \
    "$MINI_CAFFE_ROOT/src/proto/caffe.proto"
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_TOOLCHAIN_FILE \
      -DANDROID_NDK=$NDK_ROOT \
      -DANDROID_ABI=$ANDROID_ABI \
      -DANDROID_NATIVE_API_LEVEL=$ANDROID_NATIVE_API_LEVEL \
      -DCMAKE_BUILD_TYPE=Release \
      $MINI_CAFFE_ROOT
make -j$ANDROID_BUILD_JOBS
