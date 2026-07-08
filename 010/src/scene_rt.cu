// 010 | 多物体 + Phong + 硬阴影 + 镜面反射
#include "rt_math.cuh"
#include <cstdio>
#include <chrono>

// 常量内存存场景
struct Mat {
    Vec3 albedo;
    float ka, kd, ks, shininess;
    float reflect; // 镜面反射系数
};
struct Sphere { Vec3 c; float r; Mat m; };
struct Plane { Vec3 p, n; Mat m; };

__constant__ Sphere d_spheres[8];
__constant__ int d_nspheres;
__constant__ Plane d_planes[4];
__constant__ int d_nplanes;
__constant__ Vec3 d_light_pos;
__constant__ Vec3 d_light_color;

struct Hit {
    float t;
    Vec3 p, n;
    Mat m;
    bool ok;
};

__device__ bool sphere_hit(const Sphere& s, Ray r, float tmin, float tmax, Hit& h) {
    Vec3 oc = r.o - s.c;
    float a = dot(r.d, r.d);
    float half_b = dot(oc, r.d);
    float c = dot(oc, oc) - s.r * s.r;
    float disc = half_b*half_b - a*c;
    if (disc < 0) return false;
    float sq = sqrtf(disc);
    float root = (-half_b - sq) / a;
    if (root < tmin || root > tmax) {
        root = (-half_b + sq) / a;
        if (root < tmin || root > tmax) return false;
    }
    h.t = root; h.p = r.o + r.d * root; h.n = normalize(h.p - s.c); h.m = s.m; h.ok = true;
    return true;
}

__device__ bool plane_hit(const Plane& pl, Ray r, float tmin, float tmax, Hit& h) {
    float denom = dot(pl.n, r.d);
    if (fabsf(denom) < 1e-6f) return false;
    float t = dot(pl.p - r.o, pl.n) / denom;
    if (t < tmin || t > tmax) return false;
    h.t = t; h.p = r.o + r.d * t; h.n = pl.n; h.m = pl.m; h.ok = true;
    return true;
}

__device__ Hit closest_hit(Ray r, float tmin, float tmax) {
    Hit best; best.ok = false; best.t = tmax;
    Hit tmp;
    for (int i = 0; i < d_nspheres; ++i)
        if (sphere_hit(d_spheres[i], r, tmin, best.t, tmp)) best = tmp;
    for (int i = 0; i < d_nplanes; ++i)
        if (plane_hit(d_planes[i], r, tmin, best.t, tmp)) best = tmp;
    return best;
}

__device__ bool in_shadow(Vec3 p, Vec3 light) {
    // 硬阴影：从交点射向光源，中途是否有遮挡
    Vec3 dir = light - p;
    float dist = len(dir);
    Ray sray{p + normalize(dir)*1e-3f, normalize(dir)};
    Hit h = closest_hit(sray, 0.001f, dist - 0.001f);
    return h.ok;
}

__device__ Vec3 phong(Hit h, Vec3 view_dir) {
    Vec3 L = normalize(d_light_pos - h.p);
    Vec3 N = h.n;
    Vec3 V = normalize(view_dir);
    Vec3 amb = h.m.albedo * h.m.ka;
    if (in_shadow(h.p, d_light_pos)) return amb;
    float ndl = fmaxf(0.f, dot(N, L));
    Vec3 diff = h.m.albedo * h.m.kd * ndl;
    Vec3 R = reflect(-L, N);
    float spec = powf(fmaxf(0.f, dot(R, V)), h.m.shininess);
    Vec3 sp = d_light_color * h.m.ks * spec;
    return amb + diff + sp;
}

__device__ Vec3 trace(Ray r, int depth) {
    // 迭代式反射（最大 depth 次），避免设备端深递归
    Vec3 throughput{1,1,1};
    Vec3 color{0,0,0};
    for (int d = 0; d < depth; ++d) {
        Hit h = closest_hit(r, 0.001f, 1e20f);
        if (!h.ok) {
            // 天空渐变
            float t = 0.5f * (r.d.y + 1.f);
            Vec3 sky = Vec3{1,1,1}*(1-t) + Vec3{0.5f,0.7f,1.f}*t;
            color = color + throughput * sky;
            break;
        }
        Vec3 local = phong(h, -r.d);
        color = color + throughput * local * (1.f - h.m.reflect);
        if (h.m.reflect < 1e-3f) break;
        throughput = throughput * h.m.reflect;
        Vec3 refl = reflect(r.d, h.n);
        r = Ray{h.p + h.n * 1e-3f, normalize(refl)};
    }
    return color;
}

__global__ void k_render(uint8_t* out, int W, int H, int max_depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float aspect = (float)W/H;
    Vec3 cam{0, 0.8f, 2.5f};
    float u = (2.f*(x+0.5f)/W - 1.f) * aspect;
    float v = 1.f - 2.f*(y+0.5f)/H;
    Ray ray{cam, normalize(Vec3{u, v, -1.5f} - Vec3{0,0,0})};
    // 朝向原点
    ray.d = normalize(Vec3{u, v - 0.2f, -1.5f});
    Vec3 col = trace(ray, max_depth);
    // 简单 gamma
    col = Vec3{sqrtf(fminf(1.f,col.x)), sqrtf(fminf(1.f,col.y)), sqrtf(fminf(1.f,col.z))};
    int i = (y*W+x)*3;
    out[i]= (uint8_t)(col.x*255); out[i+1]=(uint8_t)(col.y*255); out[i+2]=(uint8_t)(col.z*255);
}

int main() {
    printf("============================================================\n");
    printf("010 | 多物体 + Phong + 阴影 + 反射\n");
    printf("============================================================\n");
    Sphere hs[4];
    hs[0] = {Vec3{0,0.5f,-1}, 0.5f, {{0.9f,0.2f,0.2f}, 0.1f,0.7f,0.5f,32.f, 0.3f}};
    hs[1] = {Vec3{-1.1f,0.4f,-1.5f}, 0.4f, {{0.2f,0.8f,0.3f}, 0.1f,0.6f,0.4f,16.f, 0.1f}};
    hs[2] = {Vec3{1.0f,0.35f,-0.8f}, 0.35f, {{0.9f,0.9f,0.95f}, 0.05f,0.2f,0.9f,64.f, 0.85f}}; // 镜面球
    hs[3] = {Vec3{0.2f,0.2f,-2.2f}, 0.2f, {{0.2f,0.3f,0.9f}, 0.1f,0.7f,0.3f,16.f, 0.0f}};
    Plane hp[1];
    hp[0] = {Vec3{0,0,0}, Vec3{0,1,0}, {{0.7f,0.7f,0.7f}, 0.15f,0.6f,0.2f,8.f, 0.2f}};
    int ns=4, np=1;
    Vec3 light_pos{2, 4, 2}, light_col{1,1,1};
    cudaMemcpyToSymbol(d_spheres, hs, sizeof(hs));
    cudaMemcpyToSymbol(d_nspheres, &ns, sizeof(int));
    cudaMemcpyToSymbol(d_planes, hp, sizeof(hp));
    cudaMemcpyToSymbol(d_nplanes, &np, sizeof(int));
    cudaMemcpyToSymbol(d_light_pos, &light_pos, sizeof(Vec3));
    cudaMemcpyToSymbol(d_light_color, &light_col, sizeof(Vec3));

    const int W=960, H=540;
    std::vector<uint8_t> img(W*H*3);
    uint8_t* d; cudaMalloc(&d, img.size());
    dim3 block(16,16), grid((W+15)/16,(H+15)/16);
    auto t0 = std::chrono::high_resolution_clock::now();
    k_render<<<grid,block>>>(d, W, H, 3); // 最大反射 3 次
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    cudaMemcpy(img.data(), d, img.size(), cudaMemcpyDeviceToHost);
    write_ppm("output/scene.ppm", img, W, H);
    printf("渲染 %dx%d @ 反射深度3: %.3f ms\n", W, H, ms);
    printf("✓ 多物体场景 + 阴影 + 镜面反射完成。\n");
    cudaFree(d);
    return 0;
}
