cmake_minimum_required(VERSION 3.15)
project(mono)
# 设置OpenCV的路径
set(OpenCV_DIR "/home/nio/下载/opencv-4.x/build")

find_package(OpenCV 4 REQUIRED)

set(CMAKE_CXX_STANDARD 14)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS})