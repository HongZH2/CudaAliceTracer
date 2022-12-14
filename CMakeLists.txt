cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(cuda_alice_tracer LANGUAGES CXX CUDA)
enable_language(CUDA)
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CUDA_STANDARD 11)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin/libs/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# directories
set(CUDA_ALICE_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(THIRD_PARTIES_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/third_parties)

# find OpenGL
find_package(OpenGL REQUIRED)

include_directories(
    ${CUDA_ALICE_ROOT}
    ${THIRD_PARTIES_INCLUDE}
    ${THIRD_PARTIES_INCLUDE}/imgui
    ${THIRD_PARTIES_INCLUDE}/imgui/backends
    ${THIRD_PARTIES_INCLUDE}/glfw/include
    ${OPENGL_INCLUDE_DIR}
)

# Alice Srcs
file(GLOB CUDA_UTILS utils/srcs/*.cu)
file(GLOB CUDA_CORE core/srcs/*.cu)
file(GLOB CUDA_INTERFACE interface/srcs/*)

# add library
add_library(${PROJECT_NAME} STATIC
        ${CUDA_UTILS}
        ${CUDA_CORE}
        ${CUDA_INTERFACE}
        )

target_link_libraries(
        ${PROJECT_NAME}
        ${OPENGL_LIBRARY}
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
set_target_properties( ${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(demo)





