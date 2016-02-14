mini-caffe
==========

Minimal runtime core of Caffe porting to WIN32. This repo is aimed to provide a minimal runtime of Caffe for those want to run Caffe model on Windows platform.

### What can mini-caffe do?

This repo has no CUDA, no Caffe tools which means you can only use mini-caffe to run the nerual network model in CPU mode. You should train the nerual model use caffe tools on *nix platform, mini-caffe is just an optional choice for testing the nerual model on Windows platform. If you want a fully ported Caffe, you may refer to [happynear/caffe-windows](https://github.com/happynear/caffe-windows).

### Which compiler?

VC12 in Visual Studio 2013. We only build for x86, if you know the difference, it can be easily doned with x64. What's more, We also need CMake.

### 3rdparty libraries

Since Caffe depends on many 3rdparty libraries, I have modified some code to remove the libraries Caffe use.

- no CUDA
- no DataLayer for train
- no HDF5

but we still need libraries below.

- OpenCV
- Boost
- gflags
- glog
- protobuf
- openblas

We can download pre-compiled OpenCV and Boost, and set two environment variables `OpenCV_DIR` and `Boost_DIR`. For example, `OpenCV_DIR` = `D:\3rdparty\opencv2.4.8\build` and `Boost_DIR` = `D:\3rdparty\boost_1_57_0`. Pay attention to the Compiler version and build Architecture, which will be **VC12** and **x86**.

For openblas, I already put the library in the source code.

gflags, glog, protobuf can be compiled by ourself, I add these libraries as submodules of mini-caffe. However, I also provide a pre-compiled version of these libraries. The binaries is compiled by VC12 for x86. You can download from [dropbox](https://www.dropbox.com/s/8zbimuiviiyede5/3rdparty-VC12-x86.zip?dl=0) or [baidu driver](http://pan.baidu.com/s/1hqOoCL2).

To compile these libraries yourself, you should download the source code first.

```
git submodule update --init
```

all source code are under `3rdparty/src`.

##### gflags

```
cd 3rdparty/src/gflags
mkdir build
cd build
cmake ..
```

use VS2013 to compile gflags. `Debug` and `Release`.

##### glog

glog project already provides a solution file for Visual Studio. Just compile `Debug` and `Release`.

##### protobuf

```
cd 3rdparty/src/protobuf/cmake
mkdir build
cd build
cmake .. -DBUILD_TESTING=OFF
```

use VS2013 to compile protobuf. `Debug` and `Release`.

Once you have compiled these libraries, you need collect header files and lib files to `3rdparty/include` and `3rdparty/lib`. I provide a script for collecting, just run `copydeps.bat`, it will also copy dll files to `3rdparty/bin`.

### cmake for mini-caffe

Before we use cmake to generate vs solution file, we need to use protoc.exe to generate `caffe.pb.h` and `caffe.pb.cc`. Run `generatepb.bat` will use proto.exe and copy files to include folder and source folder.

`mini-caffe.sln` is the solution file for VS2013.

### Embed mini-caffe

To use mini-caffe as a sub-project, you may refer to [mini-caffe-example](https://github.com/luoyetx/mini-caffe-example).
