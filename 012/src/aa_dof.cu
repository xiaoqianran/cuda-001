// 012 | 多重采样抗锯齿 + 薄透镜景深 + Halton
#include "rt_math.cuh"
#include <cstdio>
#include <chrono>

struct Sphere { Vec3 c; float r; Vec3 albedo; };

__device__ bool hit(Sphere s, Ray r, float& t, Vec3& n) {
    Vec3 oc=r.o-s.c; float a=dot(r.d,r.d), b=2*dot(oc,r.d), c=dot(oc,oc)-s.r*s.r;
    float d=b*b-4*a*c; if(d<0)return false; float sq=sqrtf(d);
    float root=(-b-sq)/(2*a); if(root<0.001f){root=(-b+sq)/(2*a); if(root<0.001f)return false;}
    t=root; n=normalize(r.o+r.d*t - s.c); return true;
}

// Halton 低差异序列
__device__ float halton(int index, int base) {
    float f=1.f, r=0.f; int i=index;
    while(i>0){ f/=base; r+=f*(i%base); i/=base; }
    return r;
}

__device__ Vec3 shade(Ray r) {
    Sphere scene[3] = {
        {{0,0,-1.5f},0.5f,{0.8f,0.3f,0.2f}},
        {{-0.9f,-0.1f,-2.f},0.4f,{0.2f,0.7f,0.3f}},
        {{0.8f,-0.15f,-1.8f},0.35f,{0.2f,0.3f,0.9f}},
    };
    // 地面
    float best=1e20f; Vec3 n, alb; bool ok=false;
    for(int i=0;i<3;++i){ float t; Vec3 nn; if(hit(scene[i],r,t,nn)&&t<best){best=t;n=nn;alb=scene[i].albedo;ok=true;} }
    if(r.d.y<0){ float t=- (r.o.y+0.5f)/r.d.y; if(t>0.001f&&t<best){best=t;n=Vec3{0,1,0};alb=Vec3{0.7f,0.7f,0.7f};ok=true;} }
    if(!ok){ float tt=0.5f*(r.d.y+1); return Vec3{1,1,1}*(1-tt)+Vec3{0.5f,0.7f,1}*tt; }
    float ndl=fmaxf(0.f,dot(n,normalize(Vec3{1,1,1})));
    return alb*(0.15f+0.85f*ndl);
}

// 薄透镜相机：在光圈盘采样原点，通过对焦点
__device__ Ray camera_ray(float u, float v, float lens_r, float focus_dist, int sample, bool use_halton) {
    Vec3 lookfrom{0,0.3f,1.5f};
    Vec3 lookat{0,0,-1.5f};
    Vec3 w = normalize(lookfrom-lookat);
    Vec3 uvec = normalize(cross(Vec3{0,1,0}, w));
    Vec3 vvec = cross(w, uvec);
    // 像素在焦平面上的点
    float aspect = 16.f/9.f;
    Vec3 focus = lookfrom - w*focus_dist + uvec*(u*aspect*focus_dist*0.5f) + vvec*(v*focus_dist*0.5f);
    // 光圈采样
    float rx, ry;
    if (use_halton) {
        rx = halton(sample+1, 2)*2-1; ry = halton(sample+1, 3)*2-1;
    } else {
        // LCG 伪随机
        unsigned s = (unsigned)(sample*1664525u + 1013904223u);
        rx = (s&0xFFFF)/65535.f*2-1; s = s*1664525u+1013904223u;
        ry = (s&0xFFFF)/65535.f*2-1;
    }
    // 单位圆内
    if (rx*rx+ry*ry>1){ rx*=0.7f; ry*=0.7f; }
    Vec3 origin = lookfrom + uvec*(rx*lens_r) + vvec*(ry*lens_r);
    return Ray{origin, normalize(focus - origin)};
}

__global__ void k_render(uint8_t* out, int W, int H, int spp, float lens_r, float focus, int use_halton) {
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=W||y>=H) return;
    Vec3 acc{0,0,0};
    for(int s=0;s<spp;++s){
        float ju = use_halton ? halton(s+1,2) : ((s*0.37f+0.1f)-(int)(s*0.37f+0.1f));
        float jv = (s*0.61f+0.3f)-(int)(s*0.61f+0.3f);
        // 直接内联
        ju = use_halton ? halton(s+x*W+y+1, 2) : ju;
        jv = use_halton ? halton(s+x*W+y+1, 3) : jv;
        float u = 2.f*(x+ju)/W - 1.f;
        float v = 1.f - 2.f*(y+jv)/H;
        Ray r = camera_ray(u, v, lens_r, focus, s + x + y*W, use_halton!=0);
        acc = acc + shade(r);
    }
    acc = acc * (1.f/spp);
    acc = Vec3{sqrtf(fminf(1.f,acc.x)), sqrtf(fminf(1.f,acc.y)), sqrtf(fminf(1.f,acc.z))};
    int i=(y*W+x)*3;
    out[i]=(uint8_t)(acc.x*255); out[i+1]=(uint8_t)(acc.y*255); out[i+2]=(uint8_t)(acc.z*255);
}
// 修：去掉错误引用

int main() {
    printf("============================================================\n");
    printf("012 | 抗锯齿 + 景深 (Halton/随机)\n");
    printf("============================================================\n");
    const int W=640,H=360;
    uint8_t* d; cudaMalloc(&d, W*H*3);
    dim3 b(16,16), g((W+15)/16,(H+15)/16);
    struct Cfg{const char* name; int spp; float lens; int halton;};
    Cfg cfgs[] = {
        {"spp1_pin", 1, 0.f, 0},
        {"spp16_aa", 16, 0.f, 1},
        {"spp32_dof", 32, 0.08f, 1},
        {"spp64_dof_rnd", 64, 0.08f, 0},
    };
    for (auto& c: cfgs) {
        auto t0=std::chrono::high_resolution_clock::now();
        k_render<<<g,b>>>(d,W,H,c.spp,c.lens,1.8f,c.halton);
        cudaDeviceSynchronize();
        auto t1=std::chrono::high_resolution_clock::now();
        double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
        std::vector<uint8_t> img(W*H*3);
        cudaMemcpy(img.data(),d,img.size(),cudaMemcpyDeviceToHost);
        std::string path = std::string("output/")+c.name+".ppm";
        write_ppm(path, img, W, H);
        printf("%s: spp=%d lens=%.2f -> %.2f ms\n", c.name, c.spp, c.lens, ms);
    }
    printf("✓ AA + DOF + Halton 采样完成。\n");
    return 0;
}
