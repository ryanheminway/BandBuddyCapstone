cmake_minimum_required(VERSION 3.5)
project(BandBuddy VERSION 1.0.0)

set (CMAKE_CXX_STANDARD 11)

add_definitions(-Wall -std=c++11 -g -Wextra)

add_subdirectory(common/lib)
add_subdirectory(common/tests)
add_subdirectory(network_bb/bb_msg_lib)
add_subdirectory(network_bb/server)
add_subdirectory(network_bb/tests)
add_subdirectory(big_brother)
add_subdirectory(stage1)
add_subdirectory(stage1/mock)
add_subdirectory(stage3)
add_subdirectory(stage3/mock)
