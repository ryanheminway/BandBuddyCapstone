include_directories(../network_bb/bb_msg_lib)
add_executable(big_brother big_brother.cpp big_brother_state_machine.cpp)

target_link_libraries(big_brother PRIVATE msg_lib pthread)