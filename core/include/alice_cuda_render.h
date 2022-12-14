//
// Created by zane on 12/12/22.
//

#ifndef CUDAALICETRACER_ALICE_CUDA_RENDER_CUH
#define CUDAALICETRACER_ALICE_CUDA_RENDER_CUH

#include "utils/include/alice_cuda_include.h"
#include "core/include/alice_cuda_camera.h"
#include "core/include/alice_cuda_postprocess.h"

namespace ALICE_TRACER{

    // ----------------------------
    // temporary sky box for debug
    // ----------------------------
    static __device__ glm::vec3 skybox(const Ray & r) {
        glm::vec3 dir = glm::normalize(r.dir_);
        float t = 0.5f*(dir.y + 1.0f);
        return (1.0f - t) * glm::vec3(1.0f) + t * glm::vec3(0.3f, 0.6f, 1.0f);
    }


    // ----------------------------
    // define the Ray structure
    // ----------------------------
    __global__ void render(float * img, Camera * camera);

}


#endif //CUDAALICETRACER_ALICE_CUDA_RENDER_CUH
