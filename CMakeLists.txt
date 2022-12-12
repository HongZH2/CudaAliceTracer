cmake_minimum_required(VERSION 3.23)
project(CudaAliceTracer CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include(CheckLanguage)
check_language(CUDA)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin/libs/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CUDA_ALICE_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

set(THIRD_PARTIES_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/third_parties)

include_directories(
    ${CUDA_ALICE_ROOT}
    ${THIRD_PARTIES_INCLUDE}
)

file(GLOB CUDA_UTILS_SRC utils/*)

add_executable(CudaAliceTracer demo/render.cu)

set_target_properties(CudaAliceTracer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
