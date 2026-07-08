// 008 | cuFFT 频域滤波：理想/巴特沃斯/高斯 低通/高通/带通
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdint>
#include <complex>

#define CHECK(call) do { cudaError_t e=(call); if(e){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(1);} } while(0)
#define CUFFT(call) do { cufftResult r=(call); if(r!=CUFFT_SUCCESS){fprintf(stderr,"cuFFT err %d\n",(int)r);exit(1);} } while(0)

static void write_pgm(const char* path, const float* g, int w, int h) {
    FILE* f = fopen(path, "wb");
    fprintf(f, "P5\n%d %d\n255\n", w, h);
    for (int i = 0; i < w*h; ++i) {
        float v = g[i]; if (v < 0) v = 0; if (v > 255) v = 255;
        fputc((int)(v+0.5f), f);
    }
    fclose(f);
}

// 滤波器类型
enum FilterKind { IDEAL=0, BUTTERWORTH=1, GAUSSIAN=2 };
enum PassType { LOW=0, HIGH=1, BAND=2 };

// 频域中心化后的距离
__device__ inline float freq_dist(int u, int v, int W, int H) {
    // 当前是未 shift 的 FFT 布局：低频在四角
    int du = u > W/2 ? u - W : u;
    int dv = v > H/2 ? v - H : v;
    return sqrtf((float)(du*du + dv*dv));
}

__global__ void apply_filter_kernel(cufftComplex* F, int W, int H,
                                    int kind, int pass, float d0, float d1, int order) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= W || v >= H) return;
    float D = freq_dist(u, v, W, H);
    float Hval = 1.f;
    if (kind == IDEAL) {
        if (pass == LOW) Hval = (D <= d0) ? 1.f : 0.f;
        else if (pass == HIGH) Hval = (D >= d0) ? 1.f : 0.f;
        else Hval = (D >= d0 && D <= d1) ? 1.f : 0.f;
    } else if (kind == BUTTERWORTH) {
        float eps = 1e-6f;
        if (pass == LOW) Hval = 1.f / (1.f + powf(D / (d0+eps), 2*order));
        else if (pass == HIGH) Hval = 1.f / (1.f + powf((d0+eps) / (D+eps), 2*order));
        else {
            // 带通：低通(d1) - 低通(d0)
            float Hl = 1.f / (1.f + powf(D / (d1+eps), 2*order));
            float Hh = 1.f / (1.f + powf(D / (d0+eps), 2*order));
            Hval = Hl - Hh;
        }
    } else { // GAUSSIAN
        if (pass == LOW) Hval = expf(-(D*D) / (2.f * d0 * d0));
        else if (pass == HIGH) Hval = 1.f - expf(-(D*D) / (2.f * d0 * d0));
        else {
            Hval = expf(-(D*D)/(2.f*d1*d1)) - expf(-(D*D)/(2.f*d0*d0));
        }
    }
    int idx = v * W + u;
    F[idx].x *= Hval;
    F[idx].y *= Hval;
}

__global__ void spectrum_vis_kernel(const cufftComplex* F, float* vis, int W, int H) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= W || v >= H) return;
    // fftshift 可视化
    int su = (u + W/2) % W;
    int sv = (v + H/2) % H;
    float re = F[v*W+u].x, im = F[v*W+u].y;
    float mag = log1pf(sqrtf(re*re + im*im));
    vis[sv*W + su] = mag;
}

int main() {
    printf("============================================================\n");
    printf("008 | cuFFT 频域滤波\n");
    printf("============================================================\n");
    const int W = 512, H = 512;
    std::vector<float> h_img(W*H);
    // 合成：低频渐变 + 高频棋盘
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float base = 80.f + 40.f * sinf(x * 0.02f) * cosf(y * 0.02f);
            float hi = (((x/8)+(y/8))%2) ? 40.f : 0.f;
            h_img[y*W+x] = base + hi;
        }

    cufftComplex *d_data = nullptr;
    CHECK(cudaMalloc(&d_data, sizeof(cufftComplex)*W*H));
    std::vector<cufftComplex> h_c(W*H);
    for (int i = 0; i < W*H; ++i) { h_c[i].x = h_img[i]; h_c[i].y = 0; }
    CHECK(cudaMemcpy(d_data, h_c.data(), sizeof(cufftComplex)*W*H, cudaMemcpyHostToDevice));

    cufftHandle plan;
    CUFFT(cufftPlan2d(&plan, H, W, CUFFT_C2C));
    // 前向 FFT
    CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));

    // 频谱可视化
    float* d_vis = nullptr; CHECK(cudaMalloc(&d_vis, sizeof(float)*W*H));
    dim3 block(16,16), grid((W+15)/16,(H+15)/16);
    spectrum_vis_kernel<<<grid,block>>>(d_data, d_vis, W, H);
    std::vector<float> vis(W*H);
    CHECK(cudaMemcpy(vis.data(), d_vis, sizeof(float)*W*H, cudaMemcpyDeviceToHost));
    float vmax = 1e-6f; for (float v: vis) if (v>vmax) vmax=v;
    for (float& v: vis) v = v / vmax * 255.f;

    // 三种滤波器
    struct Job { const char* name; int kind; int pass; float d0; float d1; };
    Job jobs[] = {
        {"low_ideal", IDEAL, LOW, 30, 0},
        {"high_gauss", GAUSSIAN, HIGH, 20, 0},
        {"band_butter", BUTTERWORTH, BAND, 15, 60},
    };
    for (auto& j : jobs) {
        cufftComplex* d_work = nullptr;
        CHECK(cudaMalloc(&d_work, sizeof(cufftComplex)*W*H));
        CHECK(cudaMemcpy(d_work, d_data, sizeof(cufftComplex)*W*H, cudaMemcpyDeviceToDevice));
        apply_filter_kernel<<<grid,block>>>(d_work, W, H, j.kind, j.pass, j.d0, j.d1, 2);
        CUFFT(cufftExecC2C(plan, d_work, d_work, CUFFT_INVERSE));
        std::vector<cufftComplex> out(W*H);
        CHECK(cudaMemcpy(out.data(), d_work, sizeof(cufftComplex)*W*H, cudaMemcpyDeviceToHost));
        std::vector<float> img(W*H);
        float scale = 1.f / (W*H);
        for (int i = 0; i < W*H; ++i) img[i] = out[i].x * scale;
        char path[256];
        snprintf(path, sizeof(path), "output/%s.pgm", j.name);
        write_pgm(path, img.data(), W, H);
        printf("写出 %s\n", path);
        cudaFree(d_work);
    }
    write_pgm("output/input.pgm", h_img.data(), W, H);
    write_pgm("output/spectrum.pgm", vis.data(), W, H);
    printf("✓ cuFFT 低通/高通/带通 + 频谱可视化完成。\n");
    cufftDestroy(plan);
    cudaFree(d_data); cudaFree(d_vis);
    return 0;
}
