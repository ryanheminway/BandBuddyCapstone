add_library(
    shared_mem STATIC
    shared_mem.h
    shared_mem.cpp
)

target_include_directories(shared_mem PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")