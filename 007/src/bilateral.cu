// 007 | GPU Bilateral Filter（保边去噪）
// 共享内存 tile + cudaTextureObject 采样；对比高斯模糊
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <chrono>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)

// 简易 PPM 写出
static void write_ppm(const char* path, const uint8_t* rgb, int w, int h) {
    FILE* f = fopen(path, "wb");
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    fwrite(rgb, 1, w * h * 3, f);
    fclose(f);
}

// 合成测试图
static void make_image(std::vector<uint8_t>& img, int w, int h) {
    img.resize(w * h * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int i = (y * w + x) * 3;
            img[i+0] = (uint8_t)(x * 255 / w);
            img[i+1] = (uint8_t)(y * 255 / h);
            img[i+2] = 128;
            // 强边缘方块
            if (x > w/3 && x < 2*w/3 && y > h/3 && y < 2*h/3) {
                img[i+0] = 220; img[i+1] = 40; img[i+2] = 40;
            }
            // 噪声
            img[i+0] = (uint8_t)std::min(255, std::max(0, (int)img[i+0] + (rand() % 21 - 10)));
            img[i+1] = (uint8_t)std::min(255, std::max(0, (int)img[i+1] + (rand() % 21 - 10)));
            img[i+2] = (uint8_t)std::min(255, std::max(0, (int)img[i+2] + (rand() % 21 - 10)));
        }
    }
}

// -------- 高斯模糊（对比用）--------
__global__ void gaussian_kernel(const uint8_t* src, uint8_t* dst, int w, int h, int r, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float inv2s = 1.f / (2.f * sigma * sigma);
    for (int c = 0; c < 3; ++c) {
        float acc = 0.f, wsum = 0.f;
        for (int dy = -r; dy <= r; ++dy) {
            for (int dx = -r; dx <= r; ++dx) {
                int ix = min(w-1, max(0, x+dx));
                int iy = min(h-1, max(0, y+dy));
                float ww = expf(-(dx*dx + dy*dy) * inv2s);
                acc += ww * src[(iy*w+ix)*3 + c];
                wsum += ww;
            }
        }
        dst[(y*w+x)*3 + c] = (uint8_t)(acc / wsum + 0.5f);
    }
}

// -------- 双边滤波（共享内存 + 邻域）--------
// 空间高斯 * 值域高斯，保留边缘
template<int RADIUS>
__global__ void bilateral_shared_kernel(const uint8_t* src, uint8_t* dst,
                                        int w, int h, float sigma_s, float sigma_r) {
    // tile：block 16x16 + halo
    constexpr int BX = 16, BY = 16;
    constexpr int SW = BX + 2 * RADIUS;
    constexpr int SH = BY + 2 * RADIUS;
    __shared__ float sm[3][SH][SW];

    int tx = threadIdx.x, ty = threadIdx.y;
    int x = blockIdx.x * BX + tx;
    int y = blockIdx.y * BY + ty;
    int base_x = blockIdx.x * BX - RADIUS;
    int base_y = blockIdx.y * BY - RADIUS;

    // 协作加载 halo tile
    for (int sy = ty; sy < SH; sy += BY) {
        for (int sx = tx; sx < SW; sx += BX) {
            int gx = min(w-1, max(0, base_x + sx));
            int gy = min(h-1, max(0, base_y + sy));
            int idx = (gy * w + gx) * 3;
            sm[0][sy][sx] = src[idx+0];
            sm[1][sy][sx] = src[idx+1];
            sm[2][sy][sx] = src[idx+2];
        }
    }
    __syncthreads();
    if (x >= w || y >= h) return;

    float inv2s = 1.f / (2.f * sigma_s * sigma_s);
    float inv2r = 1.f / (2.f * sigma_r * sigma_r);
    int cx = tx + RADIUS, cy = ty + RADIUS;

    for (int c = 0; c < 3; ++c) {
        float center = sm[c][cy][cx];
        float acc = 0.f, wsum = 0.f;
        for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
            for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
                float v = sm[c][cy+dy][cx+dx];
                float ws = expf(-(dx*dx + dy*dy) * inv2s);
                float dr = v - center;
                float wr = expf(-(dr*dr) * inv2r);
                float ww = ws * wr;
                acc += ww * v;
                wsum += ww;
            }
        }
        dst[(y*w+x)*3 + c] = (uint8_t)(acc / wsum + 0.5f);
    }
}

// 纹理内存版本：通过 tex2D 读取（演示 texture object）
__global__ void bilateral_texture_kernel(cudaTextureObject_t tex,
                                         uint8_t* dst, int w, int h,
                                         int radius, float sigma_s, float sigma_r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float inv2s = 1.f / (2.f * sigma_s * sigma_s);
    float inv2r = 1.f / (2.f * sigma_r * sigma_r);
    // 打包 RGB 在 float4 纹理中
    float4 center = tex2D<float4>(tex, x + 0.5f, y + 0.5f);
    float cchs[3] = {center.x, center.y, center.z};
    float outc[3] = {0,0,0};
    for (int c = 0; c < 3; ++c) {
        float acc = 0.f, wsum = 0.f;
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                float4 s = tex2D<float4>(tex, x + dx + 0.5f, y + dy + 0.5f);
                float v = (c==0? s.x : (c==1? s.y : s.z));
                float ws = expf(-(dx*dx + dy*dy) * inv2s);
                float dr = v - cchs[c];
                float wr = expf(-(dr*dr) * inv2r);
                float ww = ws * wr;
                acc += ww * v;
                wsum += ww;
            }
        }
        outc[c] = acc / wsum;
    }
    int i = (y*w+x)*3;
    dst[i+0] = (uint8_t)(outc[0]+0.5f);
    dst[i+1] = (uint8_t)(outc[1]+0.5f);
    dst[i+2] = (uint8_t)(outc[2]+0.5f);
}

int main() {
    printf("============================================================\n");
    printf("007 | GPU Bilateral Filter (shared + texture)\n");
    printf("============================================================\n");
    // 4K 目标；为了演示默认用 1920x1080，可通过环境调大
    const int W = 1920, H = 1080;
    std::vector<uint8_t> h_img;
    srand(42);
    make_image(h_img, W, H);

    uint8_t *d_src = nullptr, *d_dst = nullptr, *d_gauss = nullptr;
    CHECK(cudaMalloc(&d_src, h_img.size()));
    CHECK(cudaMalloc(&d_dst, h_img.size()));
    CHECK(cudaMalloc(&d_gauss, h_img.size()));
    CHECK(cudaMemcpy(d_src, h_img.data(), h_img.size(), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((W+15)/16, (H+15)/16);

    // 高斯
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i)
        gaussian_kernel<<<grid, block>>>(d_src, d_gauss, W, H, 5, 2.0f);
    CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_g = std::chrono::duration<double, std::milli>(t1-t0).count() / 10.0;

    // 双边 shared
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i)
        bilateral_shared_kernel<5><<<grid, block>>>(d_src, d_dst, W, H, 3.0f, 30.0f);
    CHECK(cudaDeviceSynchronize());
    t1 = std::chrono::high_resolution_clock::now();
    double ms_b = std::chrono::duration<double, std::milli>(t1-t0).count() / 10.0;

    printf("分辨率: %dx%d\n", W, H);
    printf("高斯模糊:   %8.3f ms\n", ms_g);
    printf("双边(shared): %8.3f ms\n", ms_b);

    // 纹理对象路径
    // 将图像转为 float4 pitched
    size_t pitch;
    float4* d_f4 = nullptr;
    CHECK(cudaMallocPitch(&d_f4, &pitch, W * sizeof(float4), H));
    std::vector<float4> h_f4(W * H);
    for (int i = 0; i < W*H; ++i) {
        h_f4[i] = make_float4(h_img[i*3], h_img[i*3+1], h_img[i*3+2], 0.f);
    }
    CHECK(cudaMemcpy2D(d_f4, pitch, h_f4.data(), W*sizeof(float4), W*sizeof(float4), H, cudaMemcpyHostToDevice));
    cudaResourceDesc res{}; res.resType = cudaResourceTypePitch2D;
    res.res.pitch2D.devPtr = d_f4;
    res.res.pitch2D.desc = cudaCreateChannelDesc<float4>();
    res.res.pitch2D.width = W; res.res.pitch2D.height = H; res.res.pitch2D.pitchInBytes = pitch;
    cudaTextureDesc td{}; td.addressMode[0]=td.addressMode[1]=cudaAddressModeClamp;
    td.filterMode = cudaFilterModePoint; td.readMode = cudaReadModeElementType;
    td.normalizedCoords = 0;
    cudaTextureObject_t tex = 0;
    CHECK(cudaCreateTextureObject(&tex, &res, &td, nullptr));
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 5; ++i)
        bilateral_texture_kernel<<<grid, block>>>(tex, d_dst, W, H, 5, 3.0f, 30.0f);
    CHECK(cudaDeviceSynchronize());
    t1 = std::chrono::high_resolution_clock::now();
    double ms_t = std::chrono::duration<double, std::milli>(t1-t0).count() / 5.0;
    printf("双边(texture): %8.3f ms\n", ms_t);

    std::vector<uint8_t> out(h_img.size()), gout(h_img.size());
    CHECK(cudaMemcpy(out.data(), d_dst, out.size(), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(gout.data(), d_gauss, gout.size(), cudaMemcpyDeviceToHost));
    write_ppm("output/input.ppm", h_img.data(), W, H);
    write_ppm("output/bilateral.ppm", out.data(), W, H);
    write_ppm("output/gaussian.ppm", gout.data(), W, H);
    printf("已保存 output/*.ppm （对比双边 vs 高斯：边缘应更清晰）\n");
    printf("✓ Bilateral shared+texture 完成。\n");

    cudaDestroyTextureObject(tex);
    cudaFree(d_f4); cudaFree(d_src); cudaFree(d_dst); cudaFree(d_gauss);
    return 0;
}
