//
// Created by zane on 12/14/22.
//

#ifndef CUDA_ALICE_TRACER_TEXTURE_CUDA_GL_H
#define CUDA_ALICE_TRACER_TEXTURE_CUDA_GL_H

#include "utils/include/alice_cuda_image.h"

namespace ALICE_TRACER{
    /*
     * take the image buffer from cuda to GL
     */
    class Texture{
    public:
        explicit Texture(Image * d_img);
        ~Texture();
        void update(Image * d_img);
        void drawTexture();
    protected:
//        struct cudaGraphicsResource * cuda_tid_;
        uint tid_;
        uint vbo_;
        uint vao_;
        uint program_id_;
    };
}

#endif //CUDA_ALICE_TRACER_TEXTURE_CUDA_GL_H
