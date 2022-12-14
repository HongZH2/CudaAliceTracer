//
// Created by zane on 12/12/22.
//

#ifndef CUDAALICETRACER_ALICE_CUDA_CAMERA_CUH
#define CUDAALICETRACER_ALICE_CUDA_CAMERA_CUH

#include "core/include/alice_cuda_ray.h"

namespace ALICE_TRACER{

    class Camera{
    public:
        __host__ Camera();
        __host__ Camera(const glm::vec3& pos, const glm::vec3& look_at, const glm::vec2 & resolution, float fov);

        // transfer the camera coordinate to the world cooridnate
        __device__ void cameraToWorld(glm::vec3 & dir);
        __device__ void genCameraRay(glm::vec2 & uv, glm::vec2 & offset, Ray & ray);

        glm::vec3 pos_;
        glm::vec3 look_at_;
        glm::vec3 head_up_;
        glm::vec2 resolution_;
        float fov_;
    };




}


#endif //CUDAALICETRACER_ALICE_CUDA_CAMERA_CUH
