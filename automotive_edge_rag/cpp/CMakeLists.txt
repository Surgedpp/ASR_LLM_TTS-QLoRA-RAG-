cmake_minimum_required(VERSION 3.16)
project(automotive_edge_rag)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")


add_definitions(-DPYBIND11_DETAILED_ERROR_MESSAGES=0)


# Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --includes
    OUTPUT_VARIABLE PYBIND11_INCLUDES_CFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(PYBIND11_INCLUDE_DIRS_RAW ${PYBIND11_INCLUDES_CFLAGS})
separate_arguments(PYBIND11_INCLUDE_DIRS_RAW NATIVE_COMMAND "${PYBIND11_INCLUDE_DIRS_RAW}")
set(PYBIND11_INCLUDE_DIRS)
foreach(item IN LISTS PYBIND11_INCLUDE_DIRS_RAW)
    if(item MATCHES "^-I(.*)$")
        list(APPEND PYBIND11_INCLUDE_DIRS "${CMAKE_MATCH_1}")
    endif()
endforeach()

find_package(Threads REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${Python_INCLUDE_DIRS})

set(SOURCES
    edge_llm_rag_system.cpp
    query_classifier.cpp
)

set(HEADERS
    edge_llm_rag_system.h
    query_classifier.h
)

add_library(automotive_edge_rag_lib STATIC ${SOURCES} )
target_include_directories(automotive_edge_rag_lib PRIVATE ${PYBIND11_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})

set_target_properties(automotive_edge_rag_lib PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries(automotive_edge_rag_lib
    Threads::Threads
    ${CMAKE_DL_LIBS}
    pybind11::module
    ${Python3_LIBRARIES}
)

add_executable(automotive_edge_rag_demo
    demo_main.cpp
)

target_include_directories(automotive_edge_rag_demo PRIVATE ${PYBIND11_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})

target_link_libraries(automotive_edge_rag_demo
    automotive_edge_rag_lib
    Threads::Threads
    ${CMAKE_DL_LIBS}
     ${Python3_LIBRARIES}
    pybind11::module
    /usr/local/lib/libzmq_component.so  
    zmq

)

foreach(TGT IN ITEMS  automotive_edge_rag_lib automotive_edge_rag_demo)
    add_custom_command(TARGET ${TGT} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
        "${CMAKE_SOURCE_DIR}/../python"
        "$<TARGET_FILE_DIR:${TGT}>/python"
    )

    add_custom_command(TARGET ${TGT} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
        "${CMAKE_SOURCE_DIR}/../python/vector_db"
        "$<TARGET_FILE_DIR:${TGT}>/vector_db"
    )
endforeach()
