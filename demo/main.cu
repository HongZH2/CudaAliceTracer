/*
 *  First Cuda Program for Path Tracing
 *  Author: Hong Zhang
 *  Date: 2022/12/12
 */
#include <cstdio>

/*
 * Global kernel
 */
__global__ void sum(float * a, float * b, float * res, int size){
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid < size) res[tid] = a[tid] + b[tid];
}

int main() {

    // allocate the memory
    int N = 8;
    float x[8] = {1,2,3,4,5,6,7,8};
    float y[8] = {8,7,6,5,4,3,2,1};
    float * dx;
    float * dy;

    cudaMalloc(&dx, sizeof(float) * N);
    cudaMalloc(& dy, sizeof(float) * N);

    cudaMemcpy(dx, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_size{1, 1};
    dim3 thread_size;
    thread_size.x = N;
    thread_size.y = N;

    // call the kernel
    sum<<<block_size, 7>>>(dx, dy, dy, N);

    cudaMemcpy(y, dy, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("y: ");
    for(int i = 0; i < 8; ++i){
        printf("%f ", y[i]);
    }

    cudaFree(dx);
    cudaFree(dy);

    return 0;
}
