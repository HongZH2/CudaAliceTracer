//
// Created by zane on 12/13/22.
//

#include "core/include/alice_cuda_render.h"

namespace ALICE_TRACER{

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
        camera->genCameraRay(uv, offset, ray);

        assignPixel(img, p_id, ray.dir_);
    }

}
