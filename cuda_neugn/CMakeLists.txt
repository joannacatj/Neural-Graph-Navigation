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
  src/neug_model.cpp
  src/kernels.cu
)

target_include_directories(neugn_cuda PRIVATE
  include
  ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(neugn_cuda PRIVATE CUDA::cudart)
set_target_properties(neugn_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(neugn_demo PRIVATE
  include
  ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(neugn_demo PRIVATE CUDA::cudart)
set_target_properties(neugn_demo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(demo_cu PRIVATE
  include
  ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(demo_cu PRIVATE CUDA::cudart)
set_target_properties(demo_cu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
