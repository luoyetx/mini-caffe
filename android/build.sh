#!/usr/bin/env bash

set -e

if [ -z "$NDK_ROOT" ]; then
    echo "NDK_ROOT should be set for Android build"
    exit 1
fi

# set up envs
ANDROID_ROOT=`pwd`
THIRD_PARTY_ROOT=$ANDROID_ROOT/../3rdparty/src
ANDROID_TOOLCHAIN_FILE=$ANDROID_ROOT/android-cmake/android.toolchain.cmake
ANDROID_NATIVE_API_LEVEL=21
ANDROID_BUILD_JOBS=2
ANDROID_ABIS=(arm64-v8a armeabi x86 x86_64)
MINICAFFE_JNILIBS=$ANDROID_ROOT/jniLibs

echo "Android Build Root: $ANDROID_ROOT"
echo "Android NDK Root: $NDK_ROOT"
echo "Android Native API Level: $ANDROID_NATIVE_API_LEVEL"

# check host system
if [ "$(uname)" = "Darwin" ]; then
    HOST_OS=darwin
elif [ "$(expr substr $(uname -s) 1 5)" = "Linux" ]; then
    HOST_OS=linux
elif [ "$(expr substr $(uname -s) 1 10)" = "MINGW64_NT" ]; then
    HOST_OS=windows
else
    echo "Unknown OS"
    exit 1
fi

if [ "$(uname -m)" = "x86_64"  ]; then
    HOST_BIT=x86_64
else
    HOST_BIT=x86
fi

# update submodule
echo "Update git submodules"
git submodule update --init

function build_protobuf_host {
    echo "Build protobuf host"
    cd $ANDROID_ROOT
    PROTOBUF_HOST_INSTALL_ROOT=$ANDROID_ROOT/protobuf-host-build
    PROTOBUF_HOST_BUILD_ROOT=$ANDROID_ROOT/protobuf-host-build
    mkdir -p $PROTOBUF_HOST_INSTALL_ROOT
    mkdir -p $PROTOBUF_HOST_BUILD_ROOT
    cd protobuf-host-build
    cmake -DCMAKE_INSTALL_PREFIX=$PROTOBUF_HOST_INSTALL_ROOT \
          -DCMAKE_BUILD_TYPE=Release \
          -Dprotobuf_BUILD_TESTS=OFF \
          -Dprotobuf_BUILD_SHARED_LIBS=OFF \
          -G "Unix Makefiles" \
          $THIRD_PARTY_ROOT/protobuf/cmake
    make -j$ANDROID_BUILD_JOBS
    make install
}

function build_protobuf {
    echo "Build protobuf for $ANDROID_ABI"
    cd $ANDROID_ROOT
    PROTOBUF_INSTALL_ROOT=$ANDROID_ROOT/$ANDROID_ABI-install
    PROTOBUF_BUILD_ROOT=$ANDROID_ROOT/protobuf-$ANDROID_ABI-build
    mkdir -p $PROTOBUF_INSTALL_ROOT
    mkdir -p $PROTOBUF_BUILD_ROOT
    cd $PROTOBUF_BUILD_ROOT
    cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_TOOLCHAIN_FILE \
          -DANDROID_NDK=$NDK_ROOT \
          -DANDROID_ABI=$ANDROID_ABI \
          -DANDROID_NATIVE_API_LEVEL=$ANDROID_NATIVE_API_LEVEL \
          -DCMAKE_INSTALL_PREFIX=$PROTOBUF_INSTALL_ROOT \
          -DCMAKE_BUILD_TYPE=Release \
          -Dprotobuf_BUILD_TESTS=OFF \
          -Dprotobuf_BUILD_SHARED_LIBS=OFF \
          -G "Unix Makefiles" \
          $THIRD_PARTY_ROOT/protobuf/cmake
    make -j$ANDROID_BUILD_JOBS
    make install
}

function build_openblas {
    echo "Build OpenBLAS for $ANDROID_ABI"
    cd $ANDROID_ROOT
    OPENBLAS_INSTALL_ROOT=$ANDROID_ROOT/$ANDROID_ABI-install
    OPENBLAS_BUILD_ROOT=$ANDROID_ROOT/OpenBLAS-$ANDROID_ABI-build
    mkdir -p $OPENBLAS_INSTALL_ROOT
    mkdir -p $OPENBLAS_BUILD_ROOT
    echo "Copy OpenBLAS source code"
    cp -rn $THIRD_PARTY_ROOT/OpenBLAS/* $OPENBLAS_BUILD_ROOT
    # check $ANDROID_ABI
    if [ "$ANDROID_ABI" = "arm64-v8a" ]; then
        CROSS_SUFFIX=$NDK_ROOT/toolchains/aarch64-linux-android-4.9/prebuilt/$HOST_OS-$HOST_BIT/bin/aarch64-linux-android-
        SYSROOT=$NDK_ROOT/platforms/android-$ANDROID_NATIVE_API_LEVEL/arch-arm64
        TARGET=ARMV8
        BINARY=64
    elif [ "$ANDROID_ABI" = "armeabi" ]; then
        CROSS_SUFFIX=$NDK_ROOT/toolchains/arm-linux-androideabi-4.9/prebuilt/$HOST_OS-$HOST_BIT/bin/arm-linux-androideabi-
        SYSROOT=$NDK_ROOT/platforms/android-$ANDROID_NATIVE_API_LEVEL/arch-arm
        TARGET=ARMV5
        BINARY=32
    elif [ "$ANDROID_ABI" = "x86" ]; then
        CROSS_SUFFIX=$NDK_ROOT/toolchains/x86-4.9/prebuilt/$HOST_OS-$HOST_BIT/bin/i686-linux-android-
        SYSROOT=$NDK_ROOT/platforms/android-$ANDROID_NATIVE_API_LEVEL/arch-x86
        TARGET=ATOM
        BINARY=32
    elif [ "$ANDROID_ABI" = "x86_64" ]; then
        CROSS_SUFFIX=$NDK_ROOT/toolchains/x86_64-4.9/prebuilt/$HOST_OS-$HOST_BIT/bin/x86_64-linux-android-
        SYSROOT=$NDK_ROOT/platforms/android-$ANDROID_NATIVE_API_LEVEL/arch-x86_64
        TARGET=ATOM
        BINARY=64
    else
        echo "Unsupport Android ABI: $ANDROID_ABI"
        exit 1
    fi
    cd $OPENBLAS_BUILD_ROOT
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
}

function build_minicaffe {
    echo "Build Mini-Caffe"
    cd $ANDROID_ROOT
    MINICAFFE_BUILD_ROOT=$ANDROID_ROOT/MiniCaffe-$ANDROID_ABI-build
    MINICAFFE_INSTALL_ROOT=$ANDROID_ROOT/$ANDROID_ABI-install
    mkdir -p $MINICAFFE_BUILD_ROOT
    cd $MINICAFFE_BUILD_ROOT
    MINICAFFE_ROOT=$ANDROID_ROOT/..
    PROTOC=$PROTOBUF_HOST_INSTALL_ROOT/bin/protoc
    $PROTOC -I="$MINICAFFE_ROOT/src/proto" \
            --cpp_out="$MINICAFFE_ROOT/src/proto" \
            "$MINICAFFE_ROOT/src/proto/caffe.proto"
    cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_TOOLCHAIN_FILE \
          -DANDROID_NDK=$NDK_ROOT \
          -DANDROID_ABI=$ANDROID_ABI \
          -DANDROID_NATIVE_API_LEVEL=$ANDROID_NATIVE_API_LEVEL \
          -DCMAKE_BUILD_TYPE=Release \
          -DANDROID_EXTRA_LIBRARY_PATH=$ANDROID_ROOT/$ANDROID_ABI-install \
          -G "Unix Makefiles" \
          $MINICAFFE_ROOT
    make -j$ANDROID_BUILD_JOBS
}

# build protobuf for host
build_protobuf_host

for ANDROID_ABI in ${ANDROID_ABIS[@]}; do
    echo "Build for $ANDROID_ABI"
    # build protobuf
    build_protobuf
    # build OpenBLAS
    build_openblas
    # build MiniCaffe
    build_minicaffe
    # copy result
    mkdir -p $MINICAFFE_JNILIBS/$ANDROID_ABI
    cp -f $MINICAFFE_BUILD_ROOT/libcaffe.so \
          $MINICAFFE_JNILIBS/$ANDROID_ABI/libcaffe.so
done
