cmake_minimum_required(VERSION 3.16)
project(qr_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)

add_executable(qr-engine src/main.cpp src/qr.cpp)
target_link_libraries(qr-engine PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(qr-engine PRIVATE -O2 -Wall -Wextra)
