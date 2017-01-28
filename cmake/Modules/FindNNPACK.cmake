set(NNPACK_INCLUDE_SEARCH_PATHS
    /usr/include
    /usr/local/include
    /opt/NNPACK/include
    $ENV{NNPACK_ROOT}
    $ENV{NNPACK_ROOT}/include)

set(NNPACK_LIB_SEARCH_PATHS
    /lib
    /lib64
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
    /opt/NNPACK/lib
    $ENV{NNPACK_ROOT}
    $ENV{NNPACK_ROOT}/lib)

find_path(NNPACK_INCUDE_DIR NAMES nnpack.h PATHS ${NNPACK_INCLUDE_SEARCH_PATHS})
find_library(NNPACK_LIB NAMES nnpack PATHS ${NNPACK_LIB_SEARCH_PATHS})

set(NNPACK_FOUND ON)

if(NOT NNPACK_INCUDE_DIR)
  set(NNPACK_FOUND OFF)
  message(STATUS "Could not find NNPACK include. Turning NNPACK_FOUND off")
endif()

if(NOT NNPACK_LIB)
  set(NNPACK_FOUND OFF)
  message(STATUS "Could not find NNPACK lib. Turning NNPACK_FOUND off")
endif()

if(NNPACK_FOUND)
  set(HAVE_NNPACK TRUE)
  add_definitions(-DUSE_NNPACK)
  include_directories(SYSTEM ${NNPACK_INCLUDE_DIR})
  message(STATUS "Found NNPACK include: ${NNPACK_INCLUDE_DIR}")
  message(STATUS "Found NNPACK libraries: ${NNPACK_LIB}")
else(NNPACK_FOUND)
  if(NNPACK_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find NNPACK")
  endif(NNPACK_FIND_REQUIRED)
endif(NNPACK_FOUND)

mark_as_advanced(NNPACK_INCLUDE_DIR
                 NNPACK_LIB
                 NNPACK)
