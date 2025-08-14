cmake_minimum_required(VERSION 3.12)
project(zmq_component VERSION 1.0.0 LANGUAGES CXX)

find_package(PkgConfig REQUIRED)
pkg_search_module(ZMQ REQUIRED libzmq)
find_package(Threads REQUIRED)

include_directories(
    include
)

add_library(zmq_component SHARED
    src/ZmqInterface.cpp
    src/ZmqServer.cpp
    src/ZmqClient.cpp
)

target_link_libraries(zmq_component
     zmq
    Threads::Threads
)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS zmq_component DESTINATION lib)

add_executable(demo test/demo.cpp)
target_link_libraries(demo zmq_component)
