add_executable(bb_server server.cpp)
target_link_libraries(bb_server PRIVATE msg_lib shared_mem)