//
// Created by zane on 12/12/22.
//

#ifndef CUDAALICETRACER_ALICE_CUDA_RENDER_CUH
#define CUDAALICETRACER_ALICE_CUDA_RENDER_CUH

#include "utils/alice_cuda_include.h"
//#include "utils/alice_cuda_image.cuh"
#include "core/alice_cuda_camera.cuh"
#include "core/alice_cuda_postprocess.cuh"

namespace ALICE_TRACER{

    // ----------------------------
    // temporary sky box for debug
    // ----------------------------
    __device__ glm::vec3 skybox(const Ray & r) {
        glm::vec3 dir = glm::normalize(r.dir_);
        float t = 0.5f*(dir.y + 1.0f);
        return (1.0f - t) * glm::vec3(1.0f) + t * glm::vec3(0.3f, 0.6f, 1.0f);
    }

    // ----------------------------
    // define the Ray structure
    // ----------------------------
    __global__ void render(float * img, Camera * camera) {
        int max_x = camera->resolution_.x;
        int max_y = camera->resolution_.y;
        // compute the thread id and get the uv coordinates
        int id_x = threadIdx.x + blockIdx.x * blockDim.x;
        int id_y = threadIdx.y + blockIdx.y * blockDim.y;
        if(id_x >= max_x || id_y >= max_y) return;
        int p_id = id_y * max_x * 3 + id_x * 3;

        // initial the camera ray
        glm::vec2 uv = glm::vec2(id_x, id_y);
        glm::vec2 offset(0.f, 0.f);
        Ray ray;
        genCameraRay(uv, offset, camera, ray);
        glm::vec3 col = skybox(ray);

        assignPixel(img, p_id, col);
    }



}


#endif //CUDAALICETRACER_ALICE_CUDA_RENDER_CUH
