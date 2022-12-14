//
// Created by zane on 12/12/22.
//

#ifndef CUDAALICETRACER_IMAGE_H
#define CUDAALICETRACER_IMAGE_H

#include "alice_cuda_include.h"

namespace ALICE_TRACER {
    // ------------------------
    // define the Image Class
    // ------------------------
    class Image {
    public:
        __host__ Image(int w, int h, int c);
        __host__ ~Image();
        __host__ void saveImage(const std::string & path);
        __host__ inline float * getDataBuffer(){return buffer_;}

        inline int w(){return width_;}
        inline int h(){return height_;}
        inline int c(){return channel_;}
        inline float * getDataPtr(){return buffer_;}
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
