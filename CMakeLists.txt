cmake_policy(SET CMP0074 NEW)
cmake_minimum_required(VERSION 3.10)
set(CMAKE_TOOLCHAIN_FILE "C:/Users/Zzz/Documents/vcpkg/scripts/buildsystems/vcpkg.cmake")
project(fad VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)

# 1. Look for OpenCV (Make sure it's installed on your system!)
find_package(OpenCV REQUIRED)

# 2. Add your source files
add_executable(fad src/main.cpp)

# 3. Link OpenCV to your app
target_link_libraries(fad PRIVATE ${OpenCV_LIBS})

# 4. Include your own header files
target_include_directories(fad PRIVATE include)