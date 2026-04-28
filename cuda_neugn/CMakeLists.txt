cmake_minimum_required(VERSION 3.18)
project(neugn_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(neugn_cuda
  src/main.cpp
  src/tensor_io.cpp
  src/neug_model.cpp
  src/kernels.cu
)

target_include_directories(neugn_cuda PRIVATE include)
set_target_properties(neugn_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
