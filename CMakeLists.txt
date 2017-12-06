cmake_minimum_required(VERSION 2.8)
project(ParaPano)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_50;-std=c++11;-DVERBOSE")
#set(CUDA_ARCH "-arch=sm_50" CACHE STRING "Value of the NVCC -arch option.")
#SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH}")

file(GLOB  cu  *.cu *.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})  
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(include)

#set(SOURCE_FILES main.cpp filter.cpp keyPointDetector.cpp brief.cpp stitcher.cpp)
CUDA_ADD_EXECUTABLE(ParaPano main.cpp filter.cpp keyPointDetector.cpp brief.cpp stitcher.cpp Util.cpp bitarray.cpp cudaFilter.cu cudaMatcher.cu)
#add_executable(ParaPano ${SOURCE_FILES})

target_link_libraries(ParaPano ${CUDA_LIBRARIES} ${OpenCV_LIBS})
