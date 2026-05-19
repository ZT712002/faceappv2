cmake_policy(SET CMP0074 NEW)
cmake_minimum_required(VERSION 3.10)
set(CMAKE_TOOLCHAIN_FILE "C:/Users/Zzz/Documents/vcpkg/scripts/buildsystems/vcpkg.cmake")
project(fad VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)

add_executable(fad src/main.cpp)

target_link_libraries(fad PRIVATE ${OpenCV_LIBS} dlib::dlib)

target_include_directories(fad PRIVATE include)