Android Support
===============

Mini-Caffe now can be corss compiled for Android platform.

### Prerequisites

1. You need to download Android NDK [here](https://developer.android.com/ndk/downloads/index.html) first (version higher than android-ndk-15c). Or if you use [Android Studio](https://developer.android.com/studio/index.html), you can download Android NDK through SDK Manager.
2. Set up environment variable `NDK_ROOT` for Android NDK. `NDK_ROOT=/path/to/ndk`, if you use Windows, make sure to replace path separator `\` to `/`.
3. If you use Windows, you also need the GNU toolchain. Usually [tdm-gcc](http://tdm-gcc.tdragon.net/download) should be fine. Download 64bit version of tdm-gcc like `tdmgcc64-gcc-x.x.x.exe` in the download page. You also need a shell environment to run the build. Usually [Git](https://git-scm.com/downloads) for Windows ship with `Git Bash` should be fine. **IMPORTANT**, you need to copy and rename `/path/to/TDM-GCC-64/bin/mingw32-make.exe` to `/path/to/TDM-GCC-64/bin/make.exe`.

### Build

Run `build.sh` will automatically cross compile Mini-Caffe and libraries(protobuf and OpenBLAS) it relies on. The default build option is listed below, currently you can directly change them in `build.sh` file.

```
ANDROID_PLATFORM_LEVEL=21  # android native api level
ANDROID_BUILD_JOBS=4  # threads to build
ANDROID_ABIS=(arm64-v8a armeabi-v7a)  # android abi to build
```

`build.sh` is tested on Fedora 27 with android-ndk-15c, Windows 10 with tdmgcc64-gcc-5.1.0-2.exe and Android NDK r16b.

### Result

Every library will create a build folder for every ANDROID_ABI. Mini-Caffe output `libcaffe.so` will be in `jniLibs`.

### How to use libcaffe.so

I have modify [Leliana/WhatsThis](https://github.com/Leliana/WhatsThis) which uses [MXNet](https://github.com/dmlc/mxnet) as backend. Make Mini-Caffe as its backend, checkout project [luoyetx/WhatsThis](https://github.com/luoyetx/WhatsThis). What's more, you also need to checkout the Java API [here](../java).
