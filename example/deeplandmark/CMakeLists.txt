project(deeplandmark)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)
include(../../mini-caffe.cmake)

add_executable(deeplandmark landmark.hpp landmark.cpp main.cpp)
target_link_libraries(deeplandmark caffe ${OpenCV_LIBS})
