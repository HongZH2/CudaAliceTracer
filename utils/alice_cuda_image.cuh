//
// Created by zane on 12/12/22.
//

#ifndef CUDAALICETRACER_IMAGE_H
#define CUDAALICETRACER_IMAGE_H

#include "alice_cuda_include.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_write.h"

namespace ALICE_TRACER {
    /*
     * define the Image Class
     */
    class Image {
    public:
        __host__ Image(int w, int h, int c) {
            width_ = w;
            height_ = h;
            channel_ = c;
            stride_ = w;
            num_pixels_ = w * h * c;
            checkCudaErrors(cudaMallocManaged((void **)&buffer_, sizeof(float) * num_pixels_));
        }

        __host__ ~Image() {
            checkCudaErrors(cudaFree(buffer_));
        }

        __host__ inline float * getDataBuffer(){
            return buffer_;
        }

        __host__ void saveImage(const std::string & path){
            unsigned char * normalize_buf = (unsigned char*) malloc(sizeof(unsigned char) * num_pixels_);
            for(int i = 0; i < num_pixels_; ++i){
                normalize_buf[i] = buffer_[i] * 255.999f;
            }
            stbi_write_png(path.c_str(), width_, height_, channel_, normalize_buf, 0);
            free(normalize_buf);
        }
    protected:
        int width_;
        int height_;
        int stride_;
        int channel_;
        int num_pixels_;
        float * buffer_ = nullptr;  // Unified Memory, it is free to access from both the host and device
    };

}

#endif //CUDAALICETRACER_IMAGE_H
