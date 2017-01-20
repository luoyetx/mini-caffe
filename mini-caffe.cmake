# mini-caffe.cmake

include_directories(${CMAKE_CURRENT_LIST_DIR}/3rdparty/include
                    ${CMAKE_CURRENT_LIST_DIR}/3rdparty/include/openblas
                    ${CMAKE_CURRENT_LIST_DIR}/3rdparty/include/google
                    ${CMAKE_CURRENT_LIST_DIR}/include)

link_directories(${CMAKE_CURRENT_LIST_DIR}/3rdparty/lib)

set(LIBS debug libprotobufd optimized libprotobuf
         libopenblas Shlwapi)

file(GLOB CAFFE_INCLUDE_LAYERS ${CMAKE_CURRENT_LIST_DIR}/src/caffe/layers/*.hpp)
file(GLOB CAFFE_INCLUDE_UTIL ${CMAKE_CURRENT_LIST_DIR}/include/caffe/util/*.hpp)
file(GLOB CAFFE_INCLUDE_OTHER ${CMAKE_CURRENT_LIST_DIR}/include/caffe/*.hpp
                               ${CMAKE_CURRENT_LIST_DIR}/include/caffe/proto/caffe.pb.h)
file(GLOB CAFFE_SOURCE_LAYERS ${CMAKE_CURRENT_LIST_DIR}/src/caffe/layers/*.cpp)
file(GLOB CAFFE_SOURCE_UTIL ${CMAKE_CURRENT_LIST_DIR}/src/caffe/util/*.cpp)
file(GLOB CAFFE_SOURCE_OTHER ${CMAKE_CURRENT_LIST_DIR}/src/caffe/*.cpp
                             ${CMAKE_CURRENT_LIST_DIR}/src/caffe/proto/caffe.pb.cc)

source_group(include FILES ${CAFFE_INCLUDE_OTHER})
source_group(include\\layers FILES ${CAFFE_INCLUDE_LAYERS})
source_group(include\\util FILES ${CAFFE_INCLUDE_UTIL})
source_group(src FILES ${CAFFE_SOURCE_OTHER})
source_group(src\\layers FILES ${CAFFE_SOURCE_LAYERS})
source_group(src\\util FILES ${CAFFE_SOURCE_UTIL})

set(SRC ${CAFFE_INCLUDE_LAYERS} ${CAFFE_INCLUDE_UTIL} ${CAFFE_INCLUDE_OTHER}
        ${CAFFE_SOURCE_LAYERS} ${CAFFE_SOURCE_UTIL} ${CAFFE_SOURCE_OTHER})

add_library(libcaffe SHARED ${SRC})
target_link_libraries(libcaffe ${LIBS})
