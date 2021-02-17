add_executable(dummy_wav dummy_wav_to_shared_mem.cpp)

target_link_libraries(dummy_wav PRIVATE asound shared_mem)
