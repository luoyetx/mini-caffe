Mini-Caffe
==========

[![Build Status](https://travis-ci.org/luoyetx/mini-caffe.svg?branch=master)](https://travis-ci.org/luoyetx/mini-caffe)
[![Build status](https://ci.appveyor.com/api/projects/status/x9s2iajv7rtxeo3t/branch/master?svg=true)](https://ci.appveyor.com/project/luoyetx/mini-caffe/branch/master)

Minimal runtime core of Caffe. This repo is aimed to provide a minimal C++ runtime core for those want to **Forward** a Caffe model.

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

Install OpenBLAS and protobuf library through system package manager. Then build Mini-Caffe.

```
$ sudo apt install libopenblas-dev libprotobuf-dev protobuf-compiler
$ protoc -I="./src/proto" --cpp_out="./src/proto" "./src/proto/caffe.proto"
$ mkdir build
$ cd build
$ cmake ..
$ make -j4
```

### With CUDA and CUDNN

Install CUDA and CUDNN in your system, then we can compile Mini-Caffe with GPU support. Run cmake command below.

```
$ cmake .. -DUSE_CUDA=ON -DUSE_CUDNN=ON
```

### Embed Mini-Caffe

To use Mini-Caffe as a library, you may refer to [mini-caffe-example](https://github.com/luoyetx/mini-caffe-example).
