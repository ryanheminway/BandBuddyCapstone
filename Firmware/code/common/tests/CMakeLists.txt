include_directories(../../network_bb/bb_msg_lib)
add_executable(shared_mem_write test_mem_shared_write.cpp)
target_link_libraries(shared_mem_write PRIVATE shared_mem msg_lib)

add_executable(shared_mem_read test_mem_shared_read.cpp)
target_link_libraries(shared_mem_read PRIVATE shared_mem)

add_executable(shared_mem_destroy test_mem_shared_destroy.cpp)
target_link_libraries(shared_mem_destroy PRIVATE shared_mem)