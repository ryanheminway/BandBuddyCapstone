#Find alsa lib
find_package(ALSA REQUIRED)

if(!ALSA_FOUND)
message("Could not find alsa lib in your system")
endif()

include_directories(../network_bb/bb_msg_lib)
add_executable(analog2wav analog2wav.cpp)

target_link_libraries(analog2wav PRIVATE asound shared_mem msg_lib pthread)