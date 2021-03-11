add_executable(wavs2analog wavs2analog.cpp)

include_directories(../networkbb/bb_msg_lib)

target_compile_options(wavs2analog PRIVATE -g)
target_link_libraries(wavs2analog PRIVATE asound shared_mem msg_lib pthread)