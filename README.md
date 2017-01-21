Mini-Caffe
==========

[![Build Status](https://travis-ci.org/luoyetx/mini-caffe.svg?branch=master)](https://travis-ci.org/luoyetx/mini-caffe)
[![Build status](https://ci.appveyor.com/api/projects/status/x9s2iajv7rtxeo3t/branch/master?svg=true)](https://ci.appveyor.com/project/luoyetx/mini-caffe/branch/master)

Minimal runtime core of Caffe. This repo is aimed to provide a minimal runtime of Caffe for those want to run Caffe model.

### Update

- 2017/01/16. Build with x64 and remove many code.
- 2016/12/11. Mini-Caffe now only depends on OpenBLAS and protobuf.

### What can mini-caffe do?

This repo has no CUDA, no Caffe tools which means you can only use mini-caffe to run the nerual network model in CPU mode. You should train the nerual model use caffe tools on *nix platform, mini-caffe is just an optional choice for testing the nerual model on Windows platform. If you want a fully ported Caffe, you may refer to [happynear/caffe-windows](https://github.com/happynear/caffe-windows) or [Microsoft/caffe](https://github.com/Microsoft/caffe).

### Which compiler?

VC12 in Visual Studio 2013. We only build for x64, if you know the difference, it can be easily doned with x86. What's more, We also need CMake.

### 3rdparty libraries

Since Caffe depends on many 3rdparty libraries, I have modified some code to remove the libraries Caffe use.

- no CUDA
- no DataLayer for train
- no HDF5

but we still need libraries below.

- ~~OpenCV~~
- ~~Boost~~
- ~~gflags~~
- ~~glog~~
- protobuf
- openblas

~~We can download pre-compiled OpenCV and Boost, and set two environment variables `OpenCV_DIR` and `Boost_DIR`. For example, `OpenCV_DIR` = `D:\3rdparty\opencv2.4.8\build` and `Boost_DIR` = `D:\3rdparty\boost_1_57_0`. Pay attention to the Compiler version and build Architecture, which will be **VC12** and **x86**.~~

For openblas, I already put the library in the source code.

~~gflags, glog,~~ protobuf can be compiled by ourself, I add these libraries as submodules of mini-caffe.

To compile these libraries yourself, you should download the source code first.

```
git submodule update --init
```

all source code are under `3rdparty/src`.

##### protobuf

```
cd 3rdparty/src/protobuf/cmake
mkdir build
cd build
cmake .. -DBUILD_TESTING=OFF -G "Visual Studio 12 2013 Win64"
```

use VS2013 to compile protobuf. `Debug` and `Release`.

Once you have compiled these libraries, you need collect header files and lib files to `3rdparty/include` and `3rdparty/lib`. I provide a script for collecting, just run `copydeps.bat`, it will also copy dll files to `3rdparty/bin`.

### cmake for mini-caffe

Before we use cmake to generate vs solution file, we need to use protoc.exe to generate `caffe.pb.h` and `caffe.pb.cc`. Run `generatepb.bat` will use proto.exe and copy files to include folder and source folder.

`mini-caffe.sln` is the solution file for VS2013.

### Embed mini-caffe

To use mini-caffe as a sub-project, you may refer to [mini-caffe-example](https://github.com/luoyetx/mini-caffe-example).
