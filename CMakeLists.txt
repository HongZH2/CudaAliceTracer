cmake_minimum_required(VERSION 3.23)
project(CudaAliceTracer CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(CudaAliceTracer main.cu)

set_target_properties(CudaAliceTracer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
