project(mini-caffe)
cmake_minimum_required(VERSION 3.0)

find_package(OpenCV REQUIRED)
set(BOOST_DIR $ENV{BOOST_DIR})

include_directories(3rdparty/include 3rdparty/include/openblas 3rdparty/include/google)
include_directories(3rdparty/include/gtest 3rdparty/include/gflags 3rdparty/include/glog)
include_directories(include)
include_directories(${BOOST_DIR})

link_directories(${BOOST_DIR}/stage/lib)
link_directories(3rdparty/lib)

set(LIBS debug gflagsd optimized gflags
         debug gflags_nothreadsd optimized gflags_nothreads
         debug libglogd optimized libglog
         debug libprotobufd optimized libprotobuf
         debug libprotocd optimized libprotoc
         libopenblas Shlwapi)

file(GLOB SRC src/caffe/*.c* src/caffe/layers/*.c* src/caffe/util/*.c* src/caffe/proto/*.c*)
file(GLOB HD include/caffe/*.h* include/caffe/proto/*.h* include/caffe/util/*.h*)

add_library(libcaffe STATIC ${HD} ${SRC})
target_link_libraries(libcaffe ${LIBS} ${OpenCV_LIBS})

# Python Binding
include(python/python.cmake)
