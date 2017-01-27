# mini-caffe.cmake

option(USE_CUDA "Use CUDA support" OFF)
option(USE_CUDNN "Use CUDNN support" OFF)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/Cuda.cmake)

# turn on C++11
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

# include and library
if(MSVC)
  include_directories(${CMAKE_CURRENT_LIST_DIR}/3rdparty/include
                      ${CMAKE_CURRENT_LIST_DIR}/3rdparty/include/openblas
                      ${CMAKE_CURRENT_LIST_DIR}/3rdparty/include/google
                      ${CMAKE_CURRENT_LIST_DIR}/include)
  link_directories(${CMAKE_CURRENT_LIST_DIR}/3rdparty/lib)
  list(APPEND Caffe_LINKER_LIBS debug libprotobufd optimized libprotobuf
                                libopenblas)
else()
  include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
  list(APPEND Caffe_LINKER_LIBS protobuf blas)
endif()

# source file structure
file(GLOB CAFFE_INCLUDE ${CMAKE_CURRENT_LIST_DIR}/include/caffe/*.hpp)
file(GLOB CAFFE_SRC ${CMAKE_CURRENT_LIST_DIR}/src/*.hpp
                    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
                    ${CMAKE_CURRENT_LIST_DIR}/src/*.cu)
file(GLOB CAFFE_SRC_LAYERS ${CMAKE_CURRENT_LIST_DIR}/src/layers/*.hpp
                           ${CMAKE_CURRENT_LIST_DIR}/src/layers/*.cpp
                           ${CMAKE_CURRENT_LIST_DIR}/src/layers/*.cu)
file(GLOB CAFFE_SRC_UTIL ${CMAKE_CURRENT_LIST_DIR}/src/util/*.hpp
                         ${CMAKE_CURRENT_LIST_DIR}/src/util/*.cpp
                         ${CMAKE_CURRENT_LIST_DIR}/src/util/*.cu)
file(GLOB CAFFE_SRC_PROTO ${CMAKE_CURRENT_LIST_DIR}/src/proto/caffe.pb.h
                          ${CMAKE_CURRENT_LIST_DIR}/src/proto/caffe.pb.cc)
source_group(include FILES ${CAFFE_INCLUDE})
source_group(src FILES ${CAFFE_SRC})
source_group(src\\layers FILES ${CAFFE_SRC_LAYERS})
source_group(src\\util FILES ${CAFFE_SRC_UTIL})
source_group(src\\proto FILES ${CAFFE_SRC_PROTO})

# cpp code
file(GLOB CAFFE_COMPILE_CODE ${CAFFE_INCLUDE}
                             ${CAFFE_SRC}
                             ${CAFFE_SRC_LAYERS}
                             ${CAFFE_SRC_UTIL}
                             ${CAFFE_SRC_PROTO})

if(HAVE_CUDA)
  # cuda code
  file(GLOB CAFFE_CUDA_CODE ${CMAKE_CURRENT_LIST_DIR}/src/util/*.cu
                            ${CMAKE_CURRENT_LIST_DIR}/src/layers/*.cu)
  caffe_cuda_compile(CAFFE_CUDA_OBJS ${CAFFE_CUDA_CODE})
  list(APPEND CAFFE_COMPILE_CODE ${CAFFE_CUDA_OBJS})
  # cudnn code
  if(HAVE_CUDNN)
    # source file structure
    file(GLOB CAFFE_SRC_LAYERS_CUDNN ${CMAKE_CURRENT_LIST_DIR}/src/layers/cudnn/*.hpp
                                     ${CMAKE_CURRENT_LIST_DIR}/src/layers/cudnn/*.cpp
                                     ${CMAKE_CURRENT_LIST_DIR}/src/layers/cudnn/*.cu)
    source_group(src\\layers\\cudnn FILES ${CAFFE_SRC_LAYERS_CUDNN})
    # cuda code
    file(GLOB CAFFE_CUDNN_CUDA_CODE ${CMAKE_CURRENT_LIST_DIR}/src/layers/cudnn/*.cu)
    caffe_cuda_compile(CAFFE_CUDNN_OBJS ${CAFFE_CUDNN_CUDA_CODE})
    list(APPEND CAFFE_COMPILE_CODE ${CAFFE_CUDNN_OBJS})
    # cpp code
    file(GLOB CAFFR_CUDNN_CPP_CODE ${CMAKE_CURRENT_LIST_DIR}/src/layers/cudnn/*.hpp
                                   ${CMAKE_CURRENT_LIST_DIR}/src/layers/cudnn/*.cpp)
    list(APPEND CAFFE_COMPILE_CODE ${CAFFR_CUDNN_CPP_CODE})
  endif()
endif()

add_definitions(-DCAFFE_EXPORTS)
add_library(caffe SHARED ${CAFFE_COMPILE_CODE})
target_link_libraries(caffe ${Caffe_LINKER_LIBS})
