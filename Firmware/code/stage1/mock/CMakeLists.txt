add_executable(shared_mem_to_wav_file mock_stage3_sharedmemread.cpp)
add_executable(close_sharedmem close_sharedmem.cpp)
add_executable(mock_stage1 mock_stage1.cpp)

include_directories(../../network_bb/bb_msg_lib)

target_link_libraries(shared_mem_to_wav_file PRIVATE asound shared_mem)
target_link_libraries(close_sharedmem PRIVATE asound shared_mem)
target_link_libraries(mock_stage1 PRIVATE asound shared_mem msg_lib)