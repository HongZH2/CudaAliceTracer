//
// Created by zane on 12/12/22.
//

#ifndef CUDAALICETRACER_ALICE_CUDA_POSTPROCESS_H
#define CUDAALICETRACER_ALICE_CUDA_POSTPROCESS_H
#include "utils/alice_cuda_include.h"

namespace ALICE_TRACER{


    // ----------------------------
    // transfer the linear color space to gamma
    // ----------------------------
    __device__ glm::vec3 toGammaSpace(const glm::vec3 & col){
        return glm::pow(col, glm::vec3(1.f/2.2f));
    }

    // ----------------------------
    // transfer the linear gamma space to linear
    // ----------------------------
    __device__ glm::vec3 toLinearSpace(const glm::vec3 & col){
        return glm::pow(col, glm::vec3(2.2f));
    }

    // ----------------------------
    // assign the color to the final image
    // ----------------------------
    __device__ void assignPixel(float * img, const int & id, glm::vec3 & col){
        col = toGammaSpace(col);
        img[id] = col.x;
        img[id + 1] = col.y;
        img[id + 2] = col.z;
    }
}

#endif //CUDAALICETRACER_ALICE_CUDA_POSTPROCESS_H
