// 024 | 高度图地形 + 纹理 splat + 距离 LOD + 视锥剔除 + 简易水面
#include "rt_math.cuh"
#include <cstdio>
#include <vector>
#include <cmath>
#include <chrono>

__device__ float hash(int x,int z){ unsigned n=(unsigned)(x*374761393+z*668265263); n=(n^(n>>13))*1274126177; return (n&0xFFFF)/65535.f; }
__device__ float noise(float x,float z){
    int xi=(int)floorf(x), zi=(int)floorf(z); float xf=x-xi, zf=z-zi;
    float u=xf*xf*(3-2*xf), v=zf*zf*(3-2*zf);
    float a=hash(xi,zi),b=hash(xi+1,zi),c=hash(xi,zi+1),d=hash(xi+1,zi+1);
    return a+(b-a)*u + (c-a)*v + (a-b-c+d)*u*v;
}
__device__ float fbm(float x,float z){
    float a=0,f=1,amp=1; for(int i=0;i<5;++i){ a+=noise(x*f,z*f)*amp; f*=2; amp*=0.5f; } return a;
}
__device__ float height(float x,float z){ return fbm(x*0.02f,z*0.02f)*40.f - 5.f; }

// LOD：根据距离选采样步长
__device__ int lod_step(float dist){
    if(dist<50) return 1;
    if(dist<150) return 2;
    if(dist<400) return 4;
    return 8;
}

// 视锥：简化为近平面前 + 水平/垂直角度
__device__ bool in_frustum(Vec3 p, Vec3 eye, Vec3 dir, float fov, float farp){
    Vec3 to=p-eye; float d=dot(to,dir);
    if(d<1.f || d>farp) return false;
    float ang=acosf(fminf(1.f, d/len(to)));
    return ang < fov*0.6f;
}

__global__ void k_render(uint8_t* out, int W, int H, float time){
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=W||y>=H)return;
    float u=(2.f*(x+.5f)/W-1.f)*(float)W/H, v=1.f-2.f*(y+.5f)/H;
    Vec3 eye{0, 35.f, 0}, dir=normalize(Vec3{0.3f,-0.35f,1});
    // 相机朝向正交基
    Vec3 right=normalize(cross(dir,Vec3{0,1,0}));
    Vec3 up=cross(right,dir);
    Vec3 rd=normalize(dir + right*u*0.7f + up*v*0.7f);
    // 射线步进地形
    Vec3 col{0.45f,0.65f,0.95f}; // 天空
    float t=1.f; bool hit=false; Vec3 p,n;
    for(int i=0;i<256;++i){
        p = eye + rd * t;
        float h = height(p.x, p.z);
        // 水面
        float water = 2.f + 0.3f*sinf(p.x*0.2f+time)*cosf(p.z*0.15f-time);
        float dens = p.y - fmaxf(h, water);
        int step = lod_step(t);
        if(dens < 0.5f * step){
            hit=true;
            // 梯度法线
            float e=1.f;
            float hx=height(p.x+e,p.z)-height(p.x-e,p.z);
            float hz=height(p.x,p.z+e)-height(p.x,p.z-e);
            n=normalize(Vec3{-hx, 2*e, -hz});
            bool is_water = h < water && fabsf(p.y-water)<1.f;
            if(is_water){
                // 反射天空 + 深蓝
                Vec3 refl = reflect(rd, Vec3{0,1,0});
                float fres=powf(1.f-fmaxf(0.f,-rd.y),3.f);
                Vec3 water_c = Vec3{0.05f,0.15f,0.25f}*(1-fres) + Vec3{0.5f,0.7f,0.95f}*fres;
                col = water_c;
            } else {
                // 纹理 splat：沙/草/岩 按高度与坡度
                float slope = 1.f - n.y;
                Vec3 sand{0.76f,0.7f,0.5f}, grass{0.2f,0.45f,0.15f}, rock{0.4f,0.38f,0.35f}, snow{0.9f,0.92f,0.95f};
                float w_sand = fmaxf(0.f, 1.f - fabsf(h-3.f)/8.f);
                float w_grass = fmaxf(0.f, 1.f - fabsf(h-15.f)/12.f) * (1-slope);
                float w_rock = slope*2.f + fmaxf(0.f,(h-25)/20);
                float w_snow = fmaxf(0.f,(h-32)/10);
                float ws=w_sand+w_grass+w_rock+w_snow+1e-3f;
                Vec3 alb = (sand*w_sand + grass*w_grass + rock*w_rock + snow*w_snow)*(1.f/ws);
                float ndl=fmaxf(0.f,dot(n,normalize(Vec3{0.5f,1,0.3f})));
                col = alb*(0.25f+0.75f*ndl);
            }
            break;
        }
        t += fmaxf(0.5f, dens * 0.4f) * step;
        if(t>800) break;
    }
    // 雾
    float fog=1.f-expf(-t*0.002f);
    col = col*(1-fog) + Vec3{0.6f,0.7f,0.85f}*fog;
    int i=(y*W+x)*3;
    out[i]=(uint8_t)(fminf(1.f,col.x)*255);
    out[i+1]=(uint8_t)(fminf(1.f,col.y)*255);
    out[i+2]=(uint8_t)(fminf(1.f,col.z)*255);
}

int main(){
    printf("============================================================\n");
    printf("024 | 地形 LOD + splat + 水面\n");
    printf("============================================================\n");
    const int W=640,H=360;
    uint8_t* d; cudaMalloc(&d,W*H*3);
    dim3 b(16,16),g((W+15)/16,(H+15)/16);
    auto t0=std::chrono::high_resolution_clock::now();
    k_render<<<g,b>>>(d,W,H,0.5f); cudaDeviceSynchronize();
    auto t1=std::chrono::high_resolution_clock::now();
    printf("渲染: %.1f ms (目标概念 16km 场景用噪声 fbm 代替真实数据)\n",
           std::chrono::duration<double,std::milli>(t1-t0).count());
    std::vector<uint8_t> img(W*H*3); cudaMemcpy(img.data(),d,img.size(),cudaMemcpyDeviceToHost);
    write_ppm("output/terrain.ppm",img,W,H);
    printf("✓ 地形管线完成。\n");
    return 0;
}
