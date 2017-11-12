cmake_minimum_required(VERSION 2.6)
project(ParaPano)

find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(SOURCE_FILES main.cpp filter.cpp keyPointDetector.cpp)
add_executable(ParaPano ${SOURCE_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(ParaPano ${OpenCV_LIBS})