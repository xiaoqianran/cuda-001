// 016 | 运动模糊 + 体积散射（Woodcock/delta tracking 简化）
#include "rt_math.cuh"
#include <cstdio>
#include <chrono>

__device__ float rnd(unsigned& s){ s=s*1664525u+1013904223u; return (s&0xFFFFFF)/16777216.f; }

// 运动球体：中心 c0 + t*(c1-c0)，t∈[0,1] 快门
struct MovingSphere { Vec3 c0,c1; float r; Vec3 albedo; };
__constant__ MovingSphere d_ms;
// 均匀雾：sigma_t, sigma_s
__constant__ float d_sigma_t, d_sigma_s;

__device__ Vec3 sphere_center(float time) {
    return d_ms.c0 + (d_ms.c1 - d_ms.c0)*time;
}

__device__ bool hit_moving(Ray r, float time, float& t, Vec3& n){
    Vec3 c = sphere_center(time);
    Vec3 oc=r.o-c; float a=dot(r.d,r.d), b=2*dot(oc,r.d), cc=dot(oc,oc)-d_ms.r*d_ms.r;
    float disc=b*b-4*a*cc; if(disc<0)return false;
    float sq=sqrtf(disc); float root=(-b-sq)/(2*a);
    if(root<0.001f){root=(-b+sq)/(2*a); if(root<0.001f)return false;}
    t=root; n=normalize(r.o+r.d*t - c); return true;
}

// Delta tracking：在均匀介质中采样自由程
__device__ bool sample_volume(Ray r, float tmax, unsigned& s, float& t_scatter, Vec3& Tr){
    // 透射率 Tr = exp(-sigma_t * dist)
    float majorant = d_sigma_t; // 均匀介质 majorant = sigma_t
    if(majorant < 1e-6f){ Tr=Vec3{1,1,1}; return false; }
    float t = 0.f;
    while(true){
        // 自由程采样
        float xi = rnd(s);
        t += -logf(fmaxf(xi,1e-6f)) / majorant;
        if(t >= tmax){ Tr=Vec3{expf(-d_sigma_t*tmax),expf(-d_sigma_t*tmax),expf(-d_sigma_t*tmax)}; return false; }
        // 真实散射概率 sigma_s/majorant
        if(rnd(s) < d_sigma_s / majorant){
            t_scatter = t;
            float att = expf(-d_sigma_t * t);
            Tr = Vec3{att,att,att};
            return true;
        }
        // null collision 继续
    }
}

__device__ Vec3 trace(Ray r, float time, unsigned& s) {
    float t; Vec3 n;
    bool hit = hit_moving(r, time, t, n);
    float tmax = hit ? t : 20.f;
    float tsc; Vec3 Tr;
    if(sample_volume(r, tmax, s, tsc, Tr)){
        // 各向同性散射一次 + 向光源
        Vec3 p = r.o + r.d * tsc;
        Vec3 light_dir = normalize(Vec3{2,4,1} - p);
        // 相位函数 1/4pi，简化：直接乘 albedo 雾色
        float phase = 0.25f / 3.14159265f;
        Vec3 fog_alb{0.9f,0.9f,1.0f};
        // 阴影射线简化：不遮挡
        return Tr * fog_alb * phase * 8.f * fmaxf(0.f, light_dir.y);
    }
    if(hit){
        float ndl = fmaxf(0.f, dot(n, normalize(Vec3{1,1,0.5f})));
        return Tr * d_ms.albedo * (0.2f + 0.8f*ndl);
    }
    float tt=0.5f*(r.d.y+1);
    return Tr * (Vec3{1,1,1}*(1-tt)+Vec3{0.5f,0.7f,1}*tt);
}

__global__ void k_render(uint8_t* out, int W, int H, int spp){
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=W||y>=H)return;
    unsigned s=x+y*W+7;
    float aspect=(float)W/H;
    Vec3 acc{0,0,0};
    for(int i=0;i<spp;++i){
        // 快门时间随机
        float time = rnd(s);
        float ju=rnd(s), jv=rnd(s);
        float u=(2.f*(x+ju)/W-1.f)*aspect;
        float v=1.f-2.f*(y+jv)/H;
        Ray r{Vec3{0,0.2f,2.0f}, normalize(Vec3{u,v,-1.5f})};
        acc = acc + trace(r, time, s);
    }
    acc = acc*(1.f/spp);
    auto g=[](float v){return sqrtf(fminf(1.f,fmaxf(0.f,v)));};
    int i=(y*W+x)*3;
    out[i]=(uint8_t)(g(acc.x)*255); out[i+1]=(uint8_t)(g(acc.y)*255); out[i+2]=(uint8_t)(g(acc.z)*255);
}

int main(){
    printf("============================================================\n");
    printf("016 | 运动模糊 + 体积散射 (delta tracking)\n");
    printf("============================================================\n");
    MovingSphere ms{Vec3{-0.6f,0,-1.5f}, Vec3{0.6f,0,-1.5f}, 0.4f, Vec3{0.9f,0.3f,0.2f}};
    float st=0.15f, ss=0.12f;
    cudaMemcpyToSymbol(d_ms,&ms,sizeof(ms));
    cudaMemcpyToSymbol(d_sigma_t,&st,4); cudaMemcpyToSymbol(d_sigma_s,&ss,4);
    const int W=640,H=360,spp=64;
    uint8_t* d; cudaMalloc(&d,W*H*3);
    dim3 b(16,16),g((W+15)/16,(H+15)/16);
    auto t0=std::chrono::high_resolution_clock::now();
    k_render<<<g,b>>>(d,W,H,spp); cudaDeviceSynchronize();
    auto t1=std::chrono::high_resolution_clock::now();
    printf("%dx%d spp=%d: %.1f ms\n",W,H,spp, std::chrono::duration<double,std::milli>(t1-t0).count());
    std::vector<uint8_t> img(W*H*3);
    cudaMemcpy(img.data(),d,img.size(),cudaMemcpyDeviceToHost);
    write_ppm("output/volume_mb.ppm", img, W, H);
    printf("✓ 运动模糊 + 体积散射完成。\n");
    return 0;
}
