# mini-caffe.cmake

include_directories(${CMAKE_CURRENT_LIST_DIR}/3rdparty/include
                    ${CMAKE_CURRENT_LIST_DIR}/3rdparty/include/openblas
                    ${CMAKE_CURRENT_LIST_DIR}/3rdparty/include/google
                    ${CMAKE_CURRENT_LIST_DIR}/include)

link_directories(${CMAKE_CURRENT_LIST_DIR}/3rdparty/lib)

set(LIBS debug libprotobufd optimized libprotobuf
         libopenblas Shlwapi)

file(GLOB CAFFE_INCLUDE ${CMAKE_CURRENT_LIST_DIR}/include/caffe/*.hpp)
file(GLOB CAFFE_SOURCE_LAYERS ${CMAKE_CURRENT_LIST_DIR}/src/caffe/layers/*.hpp
                              ${CMAKE_CURRENT_LIST_DIR}/src/caffe/layers/*.cpp)
file(GLOB CAFFE_SOURCE_UTIL ${CMAKE_CURRENT_LIST_DIR}/src/caffe/util/*.hpp
                            ${CMAKE_CURRENT_LIST_DIR}/src/caffe/util/*.cpp)
file(GLOB CAFFE_SOURCE_OTHER ${CMAKE_CURRENT_LIST_DIR}/src/caffe/*.hpp
                             ${CMAKE_CURRENT_LIST_DIR}/src/caffe/*.cpp)
file(GLOB CAFFE_SOURCE_PROTO ${CMAKE_CURRENT_LIST_DIR}/src/caffe/proto/caffe.pb.h
                             ${CMAKE_CURRENT_LIST_DIR}/src/caffe/proto/caffe.pb.cc)

source_group(include FILES ${CAFFE_INCLUDE})
source_group(src FILES ${CAFFE_SOURCE_OTHER})
source_group(src\\layers FILES ${CAFFE_SOURCE_LAYERS})
source_group(src\\util FILES ${CAFFE_SOURCE_UTIL})
source_group(src\\proto FILES ${CAFFE_SOURCE_PROTO})

set(SRC ${CAFFE_INCLUDE} ${CAFFE_SOURCE_PROTO}
        ${CAFFE_SOURCE_LAYERS} ${CAFFE_SOURCE_UTIL} ${CAFFE_SOURCE_OTHER})

add_definitions(-DCAFFE_EXPORTS)
add_library(libcaffe SHARED ${SRC})
target_link_libraries(libcaffe ${LIBS})
