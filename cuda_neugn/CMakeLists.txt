cmake_minimum_required(VERSION 3.18)
project(neugn_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(neugn_cuda
  src/main.cpp
  src/tensor_io.cpp
  src/neug_model.cpp
  src/kernels.cu
)

add_executable(neugn_demo
  src/demo.cu
  src/tensor_io.cpp
  src/neug_model.cpp
  src/kernels.cu
)

add_executable(demo_cu
  demo.cu
  src/tensor_io.cpp
  src/kernels.cu
  src/device_graph.cu
  src/device_neugn.cu
  src/gpu_matcher.cu
)

foreach(tgt neugn_cuda neugn_demo demo_cu)
  target_include_directories(${tgt} PRIVATE include ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(${tgt} PRIVATE CUDA::cudart)
  set_target_properties(${tgt} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()
