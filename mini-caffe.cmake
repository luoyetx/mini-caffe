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

link_directories(${BOOST_DIR}/stage/lib
                 ${CMAKE_CURRENT_LIST_DIR}/3rdparty/lib)

set(LIBS debug gflagsd optimized gflags
         debug gflags_nothreadsd optimized gflags_nothreads
         debug libglogd optimized libglog
         debug libprotobufd optimized libprotobuf
         debug libprotocd optimized libprotoc
         libopenblas Shlwapi)

file(GLOB SRC ${CMAKE_CURRENT_LIST_DIR}/src/caffe/*.c*
              ${CMAKE_CURRENT_LIST_DIR}/src/caffe/layers/*.c*
              ${CMAKE_CURRENT_LIST_DIR}/src/caffe/util/*.c*
              ${CMAKE_CURRENT_LIST_DIR}/src/caffe/proto/*.c*
              ${CMAKE_CURRENT_LIST_DIR}/include/caffe/*.h*
              ${CMAKE_CURRENT_LIST_DIR}/include/caffe/proto/*.h*
              ${CMAKE_CURRENT_LIST_DIR}/include/caffe/util/*.h*)

add_library(libcaffe STATIC ${SRC})
target_link_libraries(libcaffe ${LIBS} ${OpenCV_LIBS})
