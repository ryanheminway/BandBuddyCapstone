include_directories(../flatbuffer_messages)
include_directories(../../common/lib)

add_library(
    msg_lib STATIC
    band_buddy_msg.h
    band_buddy_msg.cpp
    band_buddy_server.h
    band_buddy_server.cpp
)

target_include_directories(msg_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")