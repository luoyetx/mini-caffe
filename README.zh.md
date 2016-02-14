mini-caffe
==========

mini-caffe 为 Windows 提供一个 Caffe 的最小运行环境。请使用 **VS2013** 编译项目，**VS2012**及以下版本不保证依赖库能够正常编译成功。亲测 VS2012 无法编译 cmake 生成的 protobuf 库。

### 安装

首先利用 git 克隆该仓库

```
git clone https://github.com/luoyetx/mini-caffe.git
cd mini-caffe
git submodule update --init --recursive
```

设置环境变量 `OpenCV_DIR` 指向 OpenCV 安装目录，比如 `D:\3rdparty\opencv2.4.8\build`。设置 `BOOST_DIR` 指向 Boost 安装目录，比如 `D:\3rdparty\boost_1_57_0`，注意如果你下载的是事先编译好的 Boost 库，请把库目录（包含有lib文件的目录改成`stage\lib`），如果你是自己源码编译的，那就不用管了。注意 Boost 库使用 32 位的。

### 编译依赖库

我裁剪了 Caffe 的源码，将大部分数据层的代码都删除了，只有 MemoryDataLayer 了，这样做可以极大地减少第三方库的依赖和编译。同时裁剪过的库只使用 CPU 模式，网络的数据层只使用内存数据。经过裁剪之后的 Caffe 只依赖如下这些库。

* OpenCV
* Boost
* gflags
* glog(部分功能没有 Windows 的实现，暴力地将 Caffe 中用到的代码注释掉了)
* protobuf
* OpenBLAS

OpenCV 和 Boost 我们一般使用事先编译好的，直接在 CMakeLists.txt 中使用，而 OpenBLAS 的库我已经直接添加在了项目的源码树中，是事先编译好的 32 库，下载连接在[这里](http://sourceforge.net/projects/openblas/files/v0.2.14/OpenBLAS-v0.2.14-Win32.zip/download)，可以视情况自行下载更换。下面介绍如何源码编译另外三个库。

##### gflags

gflags 使用了 CMake 构建工程，注意一下 CMake 的版本，如果版本过低，请去[CMake官网](http://www.cmake.org/download/)下载最新版。

```
cd 3rdparty/src/gflags
mkdir build
cd build
cmake ..
```

在利用 CMake 生成 VS 工程文件之后，直接打开工程文件编译 `Debug` 和 `Release` 两个版本。

##### glog

glog 提供了 VS 的工程文件，我们直接打开工程文件 `google-glog.sln`，因为 VS 版本的关系，可能要升级工程文件，这些都是自动的，点确定后就不必在意了。打开工程之后直接编译 `Debug` 和 `Release` 两个版本。

##### protobuf

protobuf 也使用 CMake 构建工程，注意以下 cmake 时的参数

```
cd 3rdparty/src/protobuf/cmake
mkdir build
cd build
cmake .. -DBUILD_TESTING=OFF
```

和之前一样，直接打开工程文件后编译 `Debug` 和 `Release` 两个版本。

### 编译 Caffe 前的最后准备

我们需要收集各个库的头文件，库文件和 dll 文件，我写了个 `copydeps.bat` 脚本，直接双击运行，会将所有需要的文件复制到 3rdparty 指定的目录下。

我们还需要使用 protobuf 预处理 Caffe 中的 `caffe.proto` 文件，用来生成头文件和源文件，我写了 `generatebp.bat` 脚本自动调用 `srdparty\bin\protoc.exe` 生成头文件和源文件，并将其放到指定的目录下，直接双击运行就可以了。

做完上述准备后，我们可以 cmake 来生成 Caffe 的 VS 工程文件，在源码树根目录下创建 build 目录

```
mkdir build
cd build
cmake ..
```

### 编译 Caffe

打开生成的 VS 工程文件就可以编译 Caffe 代码了，我配置了 CMakeLists.txt 生成 Caffe 静态库。

### 将 mini-caffe 作为工程的一部分

现在因为一些原因，还不能直接将上述编译生成的 `libcaffe.lib` 通过静态链接的方式加入到其他项目中，需要将 mini-caffe 的源码作为项目的一部分参与编译，我写了 `mini-caffe.cmake` 文件可以方便的将 mini-caffe 整个项目的源码作为其他项目源码的一部分，只要在相应的 CMakeLists.txt 包含这个文件即可。如下面的例子。

```
+ example
|__CMakeLists.txt
|__mini-caffe
|  |__***
|  |__mini-caffe.cmake
|  |__***
|
|__example.cpp
|__example.hpp
```

在 example 项目的 CMakeLists.txt 中加入 `include(mini-caffe/mini-caffe.cmake)` 就可以将 mini-caffe 作为项目的一部分参与编译。

具体项目结构可以参考[mini-caffe-example](https://github.com/luoyetx/mini-caffe-example)的项目配置。
