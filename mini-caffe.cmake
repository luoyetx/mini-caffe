# mini-caffe.cmake

find_package(OpenCV REQUIRED)
set(BOOST_DIR $ENV{BOOST_DIR})

include_directories(${CMAKE_CURRENT_LIST_DIR}/3rdparty/include
                    ${CMAKE_CURRENT_LIST_DIR}/3rdparty/include/openblas
                    ${CMAKE_CURRENT_LIST_DIR}/3rdparty/include/google
                    ${CMAKE_CURRENT_LIST_DIR}/3rdparty/include/gflags
                    ${CMAKE_CURRENT_LIST_DIR}/3rdparty/include/glog
                    ${CMAKE_CURRENT_LIST_DIR}/include
                    ${BOOST_DIR})

link_directories(${BOOST_DIR}/stage/lib # for self compiled
                 ${BOOST_DIR}/lib32-msvc-12.0 # for VS2013
                 ${CMAKE_CURRENT_LIST_DIR}/3rdparty/lib)

set(LIBS debug gflagsd optimized gflags
         debug gflags_nothreadsd optimized gflags_nothreads
         debug libglogd optimized libglog
         debug libprotobufd optimized libprotobuf
         libopenblas Shlwapi)

file(GLOB SRC ${CMAKE_CURRENT_LIST_DIR}/src/caffe/*.cpp
              ${CMAKE_CURRENT_LIST_DIR}/src/caffe/layers/*.cpp
              ${CMAKE_CURRENT_LIST_DIR}/src/caffe/util/*.cpp
              ${CMAKE_CURRENT_LIST_DIR}/src/caffe/proto/caffe.pb.cc
              ${CMAKE_CURRENT_LIST_DIR}/include/caffe/*.hpp
              ${CMAKE_CURRENT_LIST_DIR}/include/caffe/util/*.hpp
              ${CMAKE_CURRENT_LIST_DIR}/include/caffe/layers/*.hpp
              ${CMAKE_CURRENT_LIST_DIR}/include/caffe/proto/caffe.pb.h)

add_definitions(-DCPU_ONLY -DUSE_OPENCV)
add_library(libcaffe STATIC ${SRC})
target_link_libraries(libcaffe ${LIBS} ${OpenCV_LIBS})
