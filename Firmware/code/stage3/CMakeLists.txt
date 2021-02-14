add_executable(wavs2analog wavs2analog.cpp)

target_link_libraries(wavs2analog PRIVATE asound shared_mem)