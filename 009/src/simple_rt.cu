// 009 | 最简单 GPU 光线追踪：球体 + Lambertian
// 先 CPU 串行参考，再 CUDA 并行
#include "rt_math.cuh"
#include <chrono>
#include <cstdio>

struct Sphere {
    Vec3 center;
    float radius;
    Vec3 albedo;
};

__host__ __device__ bool hit_sphere(const Sphere& s, const Ray& r, float tmin, float tmax, float& t, Vec3& n) {
    // 光线-球体求交：|o+td - c|^2 = R^2
    Vec3 oc = r.o - s.center;
    float a = dot(r.d, r.d);
    float b = 2.f * dot(oc, r.d);
    float c = dot(oc, oc) - s.radius * s.radius;
    float disc = b*b - 4*a*c;
    if (disc < 0) return false;
    float sq = sqrtf(disc);
    float root = (-b - sq) / (2*a);
    if (root < tmin || root > tmax) {
        root = (-b + sq) / (2*a);
        if (root < tmin || root > tmax) return false;
    }
    t = root;
    Vec3 p = r.o + r.d * t;
    n = normalize(p - s.center);
    return true;
}

// CPU 串行渲染
void render_cpu(uint8_t* out, int W, int H, const Sphere& sph, Vec3 light_dir) {
    Vec3 cam{0, 0, 1.5f};
    float aspect = (float)W / H;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float u = (2.f * (x + 0.5f) / W - 1.f) * aspect;
            float v = 1.f - 2.f * (y + 0.5f) / H;
            Ray ray{cam, normalize(Vec3{u, v, -1.f})};
            float t; Vec3 n;
            Vec3 col{0.1f, 0.12f, 0.18f}; // 背景
            if (hit_sphere(sph, ray, 0.001f, 1e20f, t, n)) {
                // Lambertian：albedo * max(0, n·L)
                float ndl = fmaxf(0.f, dot(n, light_dir));
                col = sph.albedo * (0.15f + 0.85f * ndl);
            }
            int i = (y*W+x)*3;
            out[i+0] = (uint8_t)(fminf(1.f, col.x)*255);
            out[i+1] = (uint8_t)(fminf(1.f, col.y)*255);
            out[i+2] = (uint8_t)(fminf(1.f, col.z)*255);
        }
    }
}

__global__ void render_gpu(uint8_t* out, int W, int H, Sphere sph, Vec3 light_dir) {
    // 每个线程 = 一个像素
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    Vec3 cam{0, 0, 1.5f};
    float aspect = (float)W / H;
    float u = (2.f * (x + 0.5f) / W - 1.f) * aspect;
    float v = 1.f - 2.f * (y + 0.5f) / H;
    Ray ray{cam, normalize(Vec3{u, v, -1.f})};
    float t; Vec3 n;
    Vec3 col{0.1f, 0.12f, 0.18f};
    if (hit_sphere(sph, ray, 0.001f, 1e20f, t, n)) {
        float ndl = fmaxf(0.f, dot(n, light_dir));
        col = sph.albedo * (0.15f + 0.85f * ndl);
    }
    int i = (y*W+x)*3;
    out[i+0] = (uint8_t)(fminf(1.f, col.x)*255);
    out[i+1] = (uint8_t)(fminf(1.f, col.y)*255);
    out[i+2] = (uint8_t)(fminf(1.f, col.z)*255);
}

int main() {
    printf("============================================================\n");
    printf("009 | 最简单光线追踪器 (CPU → GPU)\n");
    printf("============================================================\n");
    const int W = 800, H = 600;
    Sphere sph{Vec3{0,0,-1}, 0.5f, Vec3{0.9f, 0.3f, 0.2f}};
    Vec3 L = normalize(Vec3{1, 1, 0.5f});
    std::vector<uint8_t> cpu(W*H*3), gpu(W*H*3);

    auto t0 = std::chrono::high_resolution_clock::now();
    render_cpu(cpu.data(), W, H, sph, L);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_cpu = std::chrono::duration<double, std::milli>(t1-t0).count();

    uint8_t* d_out = nullptr;
    cudaMalloc(&d_out, gpu.size());
    dim3 block(16,16), grid((W+15)/16, (H+15)/16);
    // warmup
    render_gpu<<<grid,block>>>(d_out, W, H, sph, L);
    cudaDeviceSynchronize();
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 50; ++i)
        render_gpu<<<grid,block>>>(d_out, W, H, sph, L);
    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    double ms_gpu = std::chrono::duration<double, std::milli>(t1-t0).count() / 50.0;
    cudaMemcpy(gpu.data(), d_out, gpu.size(), cudaMemcpyDeviceToHost);

    // 校验
    long long diff = 0;
    for (size_t i = 0; i < cpu.size(); ++i) diff += abs((int)cpu[i]-(int)gpu[i]);
    printf("CPU: %.3f ms | GPU: %.3f ms | 加速比: %.1fx\n", ms_cpu, ms_gpu, ms_cpu/ms_gpu);
    printf("CPU/GPU 总像素差: %lld (期望 0)\n", diff);
    write_ppm("output/cpu.ppm", cpu, W, H);
    write_ppm("output/gpu.ppm", gpu, W, H);
    printf("第一张 ray traced 图像已写出 output/gpu.ppm\n");
    printf("✓ 一像素一光线一核程 闭环完成。\n");
    cudaFree(d_out);
    return 0;
}
