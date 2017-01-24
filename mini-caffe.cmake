# mini-caffe.cmake

option(USE_CUDA "Use CUDA support" OFF)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/Utils.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/Cuda.cmake)

if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

if(MSVC)
  include_directories(${CMAKE_CURRENT_LIST_DIR}/3rdparty/include
                      ${CMAKE_CURRENT_LIST_DIR}/3rdparty/include/openblas
                      ${CMAKE_CURRENT_LIST_DIR}/3rdparty/include/google
                      ${CMAKE_CURRENT_LIST_DIR}/include)
  link_directories(${CMAKE_CURRENT_LIST_DIR}/3rdparty/lib)
  set(LIBS debug libprotobufd optimized libprotobuf
           libopenblas)
else()
  include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
  set(LIBS protobuf blas)
endif()

file(GLOB CAFFE_INCLUDE ${CMAKE_CURRENT_LIST_DIR}/include/caffe/*.hpp)
file(GLOB CAFFE_SOURCE_LAYERS ${CMAKE_CURRENT_LIST_DIR}/src/layers/*.hpp
                              ${CMAKE_CURRENT_LIST_DIR}/src/layers/*.cpp)
file(GLOB CAFFE_SOURCE_UTIL ${CMAKE_CURRENT_LIST_DIR}/src/util/*.hpp
                            ${CMAKE_CURRENT_LIST_DIR}/src/util/*.cpp)
file(GLOB CAFFE_SOURCE_OTHER ${CMAKE_CURRENT_LIST_DIR}/src/*.hpp
                             ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
file(GLOB CAFFE_SOURCE_PROTO ${CMAKE_CURRENT_LIST_DIR}/src/proto/caffe.pb.h
                             ${CMAKE_CURRENT_LIST_DIR}/src/proto/caffe.pb.cc)
file(GLOB CAFFE_CUDA_LAYERS ${CMAKE_CURRENT_LIST_DIR}/src/layers/*.cu)
file(GLOB CAFFE_CUDA_UTIL ${CMAKE_CURRENT_LIST_DIR}/src/util/*.cu)

source_group(include FILES ${CAFFE_INCLUDE})
source_group(src FILES ${CAFFE_SOURCE_OTHER})
source_group(src\\layers FILES ${CAFFE_SOURCE_LAYERS} ${CAFFE_CUDA_LAYERS})
source_group(src\\util FILES ${CAFFE_SOURCE_UTIL} ${CAFFE_CUDA_UTIL})
source_group(src\\proto FILES ${CAFFE_SOURCE_PROTO})

set(SRC ${CAFFE_INCLUDE} ${CAFFE_SOURCE_PROTO}
        ${CAFFE_SOURCE_LAYERS} ${CAFFE_SOURCE_UTIL} ${CAFFE_SOURCE_OTHER})

if(HAVE_CUDA)
  caffe_cuda_compile(CUDA_OBJS ${CAFFE_CUDA_LAYERS} ${CAFFE_CUDA_UTIL})
  list(APPEND SRC ${CUDA_OBJS})
  list(APPEND LIBS ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
endif()

add_definitions(-DCAFFE_EXPORTS)
add_library(caffe SHARED ${SRC})
target_link_libraries(caffe ${LIBS})
