/*
 *  First Cuda Program for Path Tracing
 *  Author: Hong Zhang
 *  Date: 2022/12/12
 */


#include "utils/include/alice_cuda_image.h"
#include "core/include/alice_cuda_render.h"
#include "core/include/alice_cuda_camera.h"

#include "interface/include/window.h"
#include "interface/include/imgui_widgets.h"
#include "interface/include/texture_cuda_gl.h"

int main() {
    // create an empty image
    int img_w = 1200;
    int img_h = 800;
    int img_c = 3;
    glm::vec2 resolution{img_w, img_h};

    // 1. create Window
    ALICE_TRACER::Window window{};
    window.initWindow(img_w, img_h);
    ALICE_TRACER::ImGUIWidget widgets;
    widgets.initImGui();

    auto * res_img = new ALICE_TRACER::Image(img_w, img_h, img_c);
    ALICE_TRACER::Texture texture{res_img};

    // start the timer
    clock_t start, stop;
    start = clock();

    int tile_x = 16;
    int tile_y = 16;
    dim3 blocks(img_w/tile_x + 1, img_h/tile_y + 1);
    dim3 threads(tile_x, tile_y);

    // camera
    ALICE_TRACER::Camera * camera;
    cudaMallocManaged(&camera, sizeof(ALICE_TRACER::Camera));
    camera->pos_ = glm::vec3(0.f, 0.f, 4.f);
    camera->look_at_ = glm::vec3(0.f);
    camera->head_up_ = glm::vec3(0.f, 1.f, 0.f);
    camera->resolution_ = resolution;
    camera->fov_ = glm::radians(60.f);


    ALICE_TRACER::render<<<blocks,threads>>>(res_img->getDataBuffer(), camera);
    ALICE_TRACER::checkCudaErrors(cudaGetLastError());
    ALICE_TRACER::checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();

    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "the current frame took " << timer_seconds << " seconds.\n";

    // render to the screen
    while(window.updateWindow()){
        // render to the screen
        texture.update(res_img);
        texture.drawTexture();
        widgets.updateImGui();
        window.swapBuffer();
    }
    widgets.destroyImGui();
    window.releaseWindow();

    res_img->saveImage("../showcases/test.png");
    delete res_img;
    return 0;
}
