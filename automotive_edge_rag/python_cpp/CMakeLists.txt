 cmake_minimum_required(VERSION 3.16)
project(simple_python_binding)
add_definitions(-DPYBIND11_DETAILED_ERROR_MESSAGES=0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Try to locate pybind11 headers via pip (works with Python 3.12)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --includes
    OUTPUT_VARIABLE PYBIND11_INCLUDES_CFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
# Convert "-I... -I..." into a CMake list of include dirs
set(PYBIND11_INCLUDE_DIRS_RAW ${PYBIND11_INCLUDES_CFLAGS})
separate_arguments(PYBIND11_INCLUDE_DIRS_RAW NATIVE_COMMAND "${PYBIND11_INCLUDE_DIRS_RAW}")
set(PYBIND11_INCLUDE_DIRS)
foreach(item IN LISTS PYBIND11_INCLUDE_DIRS_RAW)
    if(item MATCHES "^-I(.*)$")
        list(APPEND PYBIND11_INCLUDE_DIRS "${CMAKE_MATCH_1}")
    endif()
endforeach()

# Build options
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra)
endif()

add_executable(persistent_search_cli persistent_search_cli.cpp)

# Include dirs: pybind11 headers and Python headers
target_include_directories(persistent_search_cli PRIVATE ${PYBIND11_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})

# Link Python (pybind11 is header-only)
target_link_libraries(persistent_search_cli PRIVATE ${Python3_LIBRARIES})

# Output dir
set_target_properties(persistent_search_cli PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
foreach(TGT IN ITEMS  persistent_search_cli )
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


# Install
install(TARGETS   persistent_search_cli
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/../python/
    DESTINATION share/simple_search/python
    FILES_MATCHING PATTERN "*.py"
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/../python/vector_db/
    DESTINATION share/simple_search/vector_db
)