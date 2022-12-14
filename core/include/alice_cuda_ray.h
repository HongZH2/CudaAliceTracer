//
// Created by zane on 12/12/22.
//

#ifndef CUDAALICETRACER_ALICE_CUDA_RAY_CUH
#define CUDAALICETRACER_ALICE_CUDA_RAY_CUH

#include "utils/include/alice_cuda_include.h"

namespace ALICE_TRACER{

    // ----------------------------
    // define the Ray structure
    // ----------------------------
    class Ray {
    public:
        __device__ Ray(){};
        __device__ Ray(const glm::vec3 & o, const glm::vec3 & d, float time) {
            origin_ = o;
            dir_ = d;
            time_ = time;
        }
        inline __device__ glm::vec3 origin() const       {
            return origin_;
        }
        inline __device__ glm::vec3 direction() const    {
            return dir_;
        }
        inline __device__ glm::vec3 rayAtPoint(float t) const {
            return  origin_ + dir_ * t;
        }
        glm::vec3 origin_;
        glm::vec3 dir_;
        float time_;
    };
}


#endif //CUDAALICETRACER_ALICE_CUDA_RAY_CUH
