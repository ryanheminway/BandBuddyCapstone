include_directories(../networkbb/bb_msg_lib)

add_executable(big_brother big_brother.cpp big_brother_state_machine.cpp)

target_link_libraries(big_brother PRIVATE shared_mem msg_lib pthread)