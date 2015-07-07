mini-caffe
==========

minimal runtime core of Caffe porting to WIN32.

```
git clone https://github.com/luoyetx/mini-caffe.git
git submodule update --init --recursive
```

### Build 3rdparty deps

Since I have removed unnecessary layers from Caffe, We only need three deps to compile plus openblas, which we'll use pre-compiled version from [here](http://www.openblas.net/).

##### gflags

We first generate Config files for VS.

```
cd 3rdparty/src/gflags
mkdir build && cd build
cmake ..
```

Open `gflags.sln` and compile `debug` and `release` veresion.

##### glog

There already has a `3rdparty/src/glog/google-glog.sln` file for VS. Just open it and compile with `debug` and `release` version.

##### protobuf

generate config files first.

```
cd 3rdparty/src/protobuf/cmake
mkdir build && cd build
cmake ..
```

open `protobuf.sln` and compile with `debug` and `release` version.

### Collect deps files

We need header files and libraries file to compile the static Caffe libraries, We also need dll files to run `exe` linked with static Caffe lib.

##### gflags

copy `3rdparty/src/gflags/build/include/gflags` to `3rdparty/include/gflags`

copy `3rdparty/src/gflags/build/lib/Debug/gflags.lib` to `3rdparty/lib/gflagsd.lib`

copy `3rdparty/src/gflags/build/lib/Debug/gflags_nothreads.lib` to `3rdparty/lib/gflags_nothreadsd.lib`

copy `3rdparty/src/gflags/build/lib/Release/gflags.lib` to `3rdparty/lib/gflags.lib`

copy `3rdparty/src/gflags/build/lib/Release/gflags_nothreads.lib` to `3rdparty/lib/gflags_nothreads.lib`

##### glog

copy `3rdparty/src/glog/src/windows/glog` to `3rdparty/include/glog`

copy `3rdparty/src/glog/Debug/libglog.lib` to `3rdparty/lib/libglogd.lib`

copy `3rdparty/src/glog/Release/libglog.lib` to `3rdparty/lib/libglog.lib`

copy `3rdparty/src/glog/Release/libglog.dll` to `3rdparty/bin/libglog.dll`

##### protobuf

run `3rdparty/src/protobuf/cmake/build/extract_includes.bat` to generate headers file, then copy `3rdparty/src/protobuf/cmake/build/include/google` to `3rdparty/include/google`

copy `3rdparty/src/protobuf/cmake/build/Debug/libprotobuf.lib` to `3rdparty/lib/libprotobufd.lib`

copy `3rdparty/src/protobuf/cmake/build/Debug/libprotoc.lib` to `3rdparty/lib/libprotocd.lib`

copy `3rdparty/src/protobuf/cmake/build/Release/libprotobuf.lib` to `3rdparty/lib/libprotobuf.lib`

copy `3rdparty/src/protobuf/cmake/build/Release/libprotoc.lib` to `3rdparty/lib/libprotoc.lib`

copy `3rdparty/src/protobuf/cmake/build/Release/protoc.exe` to `3rdparty/bin/protoc.exe`

##### extra OpenBLAS

download pre-compiled 32bit libopenblas from [here](http://www.openblas.net/) and extract `include`, `lib` and `bin` to `3rdparty/include`, `3rdparty/lib`, `3rdparty/bin`

rename `3rdparty/lib/libopenblas.dll.a` to `3rdparty/lib/libopenblas.lib`

### Compile Caffe

We first need use `3rdparty/bin/proto.exe` to process `src/caffe/proto/caffe.proto`, just simply run `generatepb.bat`

Now, We can compile Caffe.

```
mkdir build && cd build
cmake ..
```

open `mini-caffe.sln` and start hacking caffe source code.
