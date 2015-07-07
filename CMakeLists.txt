project(mini-caffe)
cmake_minimum_required(VERSION 3.0)

find_package(OpenCV REQUIRED)

include_directories(3rdparty/include 3rdparty/include/openblas 3rdparty/include/google)
include_directories(3rdparty/include/gtest 3rdparty/include/gflags 3rdparty/include/glog)
include_directories(include)
include_directories("D:/3rdparty/boost_1_57_0")

link_directories("D:/3rdparty/boost_1_57_0/stage/lib")
link_directories(3rdparty/lib)

set(LIBS gflags gflags_nothreads libglog libprotobuf libprotoc libopenblas Shlwapi)

file(GLOB SRC src/caffe/*.c* src/caffe/layers/*.c* src/caffe/util/*.c* src/caffe/proto/*.c*)
file(GLOB HD include/caffe/*.h* include/caffe/proto/*.h* include/caffe/util/*.h*)

add_library(caffe STATIC ${HD} ${SRC})
target_link_libraries(caffe ${LIBS} ${OpenCV_LIBS})
