set(HAVE_PYTHON ON)
set(PYTHON_INCLUDE_DIRS $ENV{PYTHON_DIR}/include)
set(NUMPY_INCLUDE_DIR $ENV{PYTHON_DIR}/Lib/site-packages/numpy/core/include)
link_directories($ENV{PYTHON_DIR}/libs
                 $ENV{PYTHON_DIR}/Lib/site-packages/numpy/core/lib)
set(PYTHON_LIBRARIES python27 npymath.lib)

if(NOT HAVE_PYTHON)
  message(STATUS "Python interface is disabled or not all required dependecies found. Building without it...")
  return()
endif()

include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})

add_library(pycaffe SHARED python/caffe/_caffe.cpp)
target_link_libraries(pycaffe libcaffe ${PYTHON_LIBRARIES})
set_target_properties(pycaffe PROPERTIES PREFIX "" OUTPUT_NAME "_caffe")
