//
// Created by zane on 12/12/22.
//

#ifndef CUDA_ALICE_TRACER_ALICE_CUDA_INCLUDE_H
#define CUDA_ALICE_TRACER_ALICE_CUDA_INCLUDE_H

#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <string>
#include <ctime>
#include <vector>

#define GL_GLEXT_PROTOTYPES
#include "GL/gl.h"
#include "GL/glext.h"

#include <cuda.h>
#include <cuda_gl_interop.h>

#define GLM_FORCE_CUDA
#include "third_parties/glm/glm.hpp"

#define MIN_THRESHOLD 1e-4

namespace ALICE_TRACER {
    #define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

    static void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
        if (result) {
            std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                      file << ":" << line << " '" << func << "' \n";
            cudaDeviceReset();   // Make sure we call CUDA Device Reset before exiting
            exit(99);
        }
    }
}


#endif //CUDA_ALICE_TRACER_ALICE_CUDA_INCLUDE_H
