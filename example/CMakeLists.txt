project(MiniCaffe-Examples)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)
include(../mini-caffe.cmake)

add_executable(deeplandmark deeplandmark/landmark.hpp deeplandmark/landmark.cpp deeplandmark/main.cpp)
target_link_libraries(deeplandmark caffe ${OpenCV_LIBS})

add_executable(wgan wgan/main.cpp)
target_link_libraries(wgan caffe ${OpenCV_LIBS})
