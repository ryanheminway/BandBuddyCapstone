#Find alsa lib
find_package(ALSA REQUIRED)

if(!ALSA_FOUND)
message("Could not find alsa lib in your system")
endif()

add_executable(analog2wav analog2wav.cpp)

target_link_libraries(analog2wav PRIVATE asound shared_mem)