//
// Created by zane on 12/12/22.
//

#ifndef CUDAALICETRACER_ALICE_CUDA_CAMERA_CUH
#define CUDAALICETRACER_ALICE_CUDA_CAMERA_CUH

#include "core/alice_cuda_ray.cuh"

namespace ALICE_TRACER{

    class Camera{
    public:
        __host__ Camera(){};
        __host__ Camera(const glm::vec3& pos, const glm::vec3& look_at, const glm::vec2 & resolution, float fov):
            pos_(pos), look_at_(look_at), resolution_(resolution), fov_(fov){
        }

        // transfer the camera coordinate to the world cooridnate
        __device__ void cameraToWorld(glm::vec3 & dir){
            // rotate the camera ray
            glm::vec3 forward = glm::normalize(pos_ - look_at_);
            if(glm::dot(head_up_, forward) > 1.f - MIN_THRESHOLD){
                head_up_ = glm::vec3 (0.f, 0.f, 1.f);
            }
            glm::vec3 right = glm::cross(head_up_, forward);
            glm::mat3 transform = {forward, right, head_up_};
            dir = transform * dir;
        }

        glm::vec3 pos_;
        glm::vec3 look_at_;
        glm::vec3 head_up_;
        glm::vec2 resolution_;
        float fov_;
    };


    __device__ void genCameraRay(glm::vec2 & uv, glm::vec2 & offset, Camera * camera, Ray & ray){
        // compute the direction of camera ray
        glm::vec2 c_pixel = uv + offset; // center of the current pixel
        glm::vec2 c_resolution = camera->resolution_ - 1.f;
        float tan_alpha = tan(camera->fov_/2.f);
        float ratio = camera->resolution_.x / camera->resolution_.y;
        glm::vec3 dir;
        dir.x = -1.f;
        dir.y = (2.f * c_pixel.x/c_resolution.x - 1.f)* ratio * tan_alpha;
        dir.z = (-2.f * c_pixel.y/c_resolution.y + 1.f) * tan_alpha;
        dir = glm::normalize(dir);
        camera->cameraToWorld(dir);
        ray.origin_ = camera->pos_;
        ray.dir_ = glm::vec3(dir);
        ray.time_ = 0.f;
    }


}


#endif //CUDAALICETRACER_ALICE_CUDA_CAMERA_CUH
