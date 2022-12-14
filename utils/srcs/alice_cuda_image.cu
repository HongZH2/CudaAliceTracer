//
// Created by zane on 12/13/22.
//
#include "utils/include/alice_cuda_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_write.h"

namespace ALICE_TRACER{
    __host__ Image::Image(int w, int h, int c) {
        width_ = w;
        height_ = h;
        channel_ = c;
        stride_ = w;
        num_pixels_ = w * h * c;
        checkCudaErrors(cudaMallocManaged((void **)&buffer_, sizeof(float) * num_pixels_));
    }

    __host__ Image::~Image() {
        checkCudaErrors(cudaFree(buffer_));
    }

    __host__ void Image::saveImage(const std::string & path){
        auto * normalize_buf = (unsigned char*) malloc(sizeof(unsigned char) * num_pixels_);
        for(int i = 0; i < num_pixels_; ++i){
            normalize_buf[i] = buffer_[i] * 255.999f;
        }
        stbi_write_png(path.c_str(), width_, height_, channel_, normalize_buf, 0);
        free(normalize_buf);
    }

}