Mini-Caffe
==========

[![Build Status](https://travis-ci.org/luoyetx/mini-caffe.svg?branch=master)](https://travis-ci.org/luoyetx/mini-caffe)
[![Build status](https://ci.appveyor.com/api/projects/status/x9s2iajv7rtxeo3t/branch/master?svg=true)](https://ci.appveyor.com/project/luoyetx/mini-caffe/branch/master)

Minimal runtime core of [Caffe](https://github.com/BVLC/caffe). This repo is aimed to provide a minimal C++ runtime core for those want to **Forward** a Caffe model.

### What can Mini-Caffe do?

Mini-Caffe only depends on OpenBLAS and protobuf which means you can't train model with Mini-Caffe. It also only supports **Forward** function which means you can't apply models like nerual art style transform that uses **Backward** function.

### Build on Windows

You need a VC compiler to build Mini-Caffe. Visual Studio 2013 Community should be fine. You can download from [here](https://www.visualstudio.com/downloads/).

##### OpenBLAS

OpenBLAS library is already shipped with the source code, we don't need to compile it. If you want, you could download other version from [here](https://sourceforge.net/projects/openblas/files/). [v0.2.14](https://sourceforge.net/projects/openblas/files/v0.2.14/) is used for Mini-Caffe.

##### protobuf

protobuf is a git submodule in Mini-Caffe, we need to fetch the source code and compile it.

```
$ git submodule update --init
$ cd 3rdparty/src/protobuf/cmake
$ mkdir build
$ cd build
$ cmake .. -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -G "Visual Studio 12 2013 Win64"
```

Use `protobuf.sln` to compile `Debug` and `Release` version.

With these two libraries, we can compile Mini-Caffe now. Copy protobuf's include headers and libraries. Generate `caffe.pb.h` and `caffe.pb.cc`.

```
$ copydeps.bat
$ generatepb.bat
$ mkdir build
$ cd build
$ cmake .. -G "Visual Studio 12 2013 Win64"
```

Use `mini-caffe.sln` to compile it.

### Build on Linux

Install OpenBLAS and protobuf library through system package manager. Or you can compile OpenBLAS and protobuf by yourself. Then build Mini-Caffe.

```
$ sudo apt install libopenblas-dev libprotobuf-dev protobuf-compiler
$ ./generatepb.sh
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release
$ make -j4
```

If you don't use Ubuntu, then you may need to install OpenBLAS and protobuf through your system package manager if any.

### Build on Mac OSX

Install OpenBLAS and protobuf library through `brew`.

```
$ brew install openblas protobuf
$ ./generate.sh
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release
$ make -j4
```

### Build for Android

Mini-Caffe now can be cross compiled for Android platform, checkout the document [here](android).

### With CUDA and CUDNN support

Install [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn) in your system, then we can compile Mini-Caffe with GPU support. Run CMake command below.

```
$ cmake .. -DUSE_CUDA=ON -DUSE_CUDNN=ON
```

Currently we only test mini-caffe on CUDA8.0 with cuDNN5.1 and CUDA9.0 with cuDNN7.1.

### With Java support

Install Java and set environment variable `JAVA_HOME`. Run CMake command below.

```
$ cmake .. -DUSE_JAVA=ON
```

### With Python support

checkout Python API [here](python), install package via `python setup.py install`.

### How to use Mini-Caffe

To use Mini-Caffe as a library, you may refer to [example](example).

### How to profile your network

The Profiler in Mini-Caffe can help you profile your network performance, see docs [here](profile.md).
