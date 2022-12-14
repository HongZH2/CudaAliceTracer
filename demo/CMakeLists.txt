cmake_minimum_required(VERSION 3.20)
project(cuda_render LANGUAGES CXX CUDA)

# imgui
set(IMGUI_INCLUDE ${THIRD_PARTIES_INCLUDE}/imgui/)

# glfw
set(GLFW_INCLUDE ${THIRD_PARTIES_INCLUDE}/glfw/include)
set(GLFW_LIB ${THIRD_PARTIES_INCLUDE}/glfw/libs/libglfw3.a)

# find OpenGL
find_package(OpenGL REQUIRED)
# find threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)


include_directories(
        ${CUDA_ALICE_ROOT}
        ${THIRD_PARTIES_INCLUDE}
        ${OPENGL_INCLUDE_DIR}
        ${GLFW_INCLUDE}
        ${IMGUI_INCLUDE}
        ${IMGUI_INCLUDE}/backends
)

# imgui srcs
file(GLOB IMGUI_SRC
        ${IMGUI_INCLUDE}/*.h
        ${IMGUI_INCLUDE}/*.cpp
        ${IMGUI_INCLUDE}/backends/imgui_impl_glfw.*
        ${IMGUI_INCLUDE}/backends/imgui_impl_opengl3*
)

add_executable(${PROJECT_NAME}
                render.cu
                ${IMGUI_SRC})

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


target_link_libraries(
        ${PROJECT_NAME}
        ${CMAKE_DL_LIBS}
        Threads::Threads
        ${OPENGL_LIBRARY}
        ${GLFW_LIB}
        cuda_alice_tracer
)



