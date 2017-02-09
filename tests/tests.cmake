# cpp
add_executable(run_net ${CMAKE_CURRENT_LIST_DIR}/run_net.cpp)
target_link_libraries(run_net caffe)

# c
add_executable(run_net_c ${CMAKE_CURRENT_LIST_DIR}/run_net.c)
target_link_libraries(run_net_c caffe)
