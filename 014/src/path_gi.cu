// 014 | AO + 漫反射 GI + 路径追踪（俄罗斯轮盘赌）
#include "rt_math.cuh"
#include <cstdio>
#include <chrono>

struct Sphere { Vec3 c; float r; Vec3 albedo; Vec3 emit; };
__constant__ Sphere d_s[8];
__constant__ int d_n;

__device__ float rnd(unsigned& s){ s=s*1664525u+1013904223u; return (s&0xFFFFFF)/16777216.f; }
__device__ Vec3 rand_hemi(Vec3 n, unsigned& s) {
    // 余弦加权半球
    float r1=rnd(s), r2=rnd(s);
    float phi=6.2831853f*r1; float cosT=sqrtf(1-r2), sinT=sqrtf(r2);
    Vec3 w=n; Vec3 a=(fabsf(w.x)>0.1f)?Vec3{0,1,0}:Vec3{1,0,0};
    Vec3 u=normalize(cross(a,w)); Vec3 v=cross(w,u);
    return normalize(u*cosf(phi)*sinT + v*sinf(phi)*sinT + w*cosT);
}

__device__ bool hit_world(Ray r, float tmin, float tmax, float& t, Vec3& n, Vec3& alb, Vec3& emit){
    bool ok=false; t=tmax;
    for(int i=0;i<d_n;++i){
        Vec3 oc=r.o-d_s[i].c; float a=dot(r.d,r.d), hb=dot(oc,r.d), c=dot(oc,oc)-d_s[i].r*d_s[i].r;
        float disc=hb*hb-a*c; if(disc<0) continue;
        float sq=sqrtf(disc); float root=(-hb-sq)/a;
        if(root<tmin||root>t){ root=(-hb+sq)/a; if(root<tmin||root>t) continue; }
        t=root; n=normalize(r.o+r.d*t - d_s[i].c); alb=d_s[i].albedo; emit=d_s[i].emit; ok=true;
    }
    // 地面
    if(fabsf(r.d.y)>1e-6f){
        float tp=(-0.5f-r.o.y)/r.d.y;
        if(tp>tmin && tp<t){ t=tp; n=Vec3{0,1,0}; alb=Vec3{0.75f,0.75f,0.75f}; emit=Vec3{0,0,0}; ok=true; }
    }
    return ok;
}

__device__ float ao_at(Vec3 p, Vec3 n, unsigned& s, int samples){
    // 环境光遮蔽：半球采样遮挡比例
    int free_cnt=0;
    for(int i=0;i<samples;++i){
        Vec3 d=rand_hemi(n,s);
        Ray ar{p+n*1e-3f, d};
        float t; Vec3 nn,alb,em;
        if(!hit_world(ar,0.001f,1.5f,t,nn,alb,em)) free_cnt++;
    }
    return free_cnt / (float)samples;
}

__device__ Vec3 path_trace(Ray r, unsigned& s, int max_depth) {
    Vec3 col{0,0,0}, thr{1,1,1};
    for(int d=0; d<max_depth; ++d){
        float t; Vec3 n,alb,em;
        if(!hit_world(r,0.001f,1e20f,t,n,alb,em)){
            col = col + thr*Vec3{0.4f,0.5f,0.7f}*0.5f; break;
        }
        Vec3 p=r.o+r.d*t;
        col = col + thr*em;
        // 漫反射弹射
        Vec3 dir=rand_hemi(n,s);
        thr = thr * alb; // Lambert /pi * cos / pdf 简化为 albedo
        r = Ray{p+n*1e-3f, dir};
        if(d>2){
            float pcont=fminf(0.95f, fmaxf(thr.x,fmaxf(thr.y,thr.z)));
            if(rnd(s)>pcont) break;
            thr = thr*(1.f/pcont);
        }
    }
    return col;
}

__global__ void k_render(uint8_t* out, int W, int H, int spp, int mode){
    // mode: 0=AO only, 1=1-bounce GI, 2=full path
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=W||y>=H)return;
    unsigned s = 1u + x + y*9973u;
    float aspect=(float)W/H;
    Vec3 acc{0,0,0};
    for(int i=0;i<spp;++i){
        float u=(2.f*(x+rnd(s))/W-1.f)*aspect;
        float v=1.f-2.f*(y+rnd(s))/H;
        Ray r{Vec3{0,0.4f,2.0f}, normalize(Vec3{u,v-0.15f,-1.8f})};
        if(mode==0){
            float t; Vec3 n,alb,em;
            if(hit_world(r,0.001f,1e20f,t,n,alb,em)){
                float ao=ao_at(r.o+r.d*t, n, s, 16);
                acc = acc + alb*ao;
            } else acc = acc + Vec3{0.5f,0.6f,0.8f};
        } else if(mode==1){
            // 一次间接光
            float t; Vec3 n,alb,em;
            if(hit_world(r,0.001f,1e20f,t,n,alb,em)){
                Vec3 p=r.o+r.d*t;
                Vec3 direct = em;
                Vec3 indir_dir=rand_hemi(n,s);
                Ray ir{p+n*1e-3f, indir_dir};
                float t2; Vec3 n2,a2,e2;
                Vec3 indir{0,0,0};
                if(hit_world(ir,0.001f,1e20f,t2,n2,a2,e2)) indir = e2 + a2*0.3f;
                else indir = Vec3{0.3f,0.35f,0.45f};
                acc = acc + direct + alb*indir;
            } else acc = acc + Vec3{0.4f,0.5f,0.7f};
        } else {
            acc = acc + path_trace(r, s, 8);
        }
    }
    acc = acc*(1.f/spp);
    auto g=[](float v){return sqrtf(fminf(1.f,fmaxf(0.f,v)));};
    int i=(y*W+x)*3;
    out[i]=(uint8_t)(g(acc.x)*255); out[i+1]=(uint8_t)(g(acc.y)*255); out[i+2]=(uint8_t)(g(acc.z)*255);
}

int main(){
    printf("============================================================\n");
    printf("014 | AO + GI + 路径追踪\n");
    printf("============================================================\n");
    Sphere hs[5]={
        {{0,0,-1.5f},0.5f,{0.8f,0.3f,0.2f},{0,0,0}},
        {{-1.1f,0,-1.8f},0.45f,{0.2f,0.7f,0.3f},{0,0,0}},
        {{1.0f,0,-1.6f},0.4f,{0.2f,0.3f,0.8f},{0,0,0}},
        {{0,1.5f,-1.2f},0.25f,{1,1,1},{12,11,9}}, // 灯
        {{-0.5f,-0.2f,-2.5f},0.3f,{0.9f,0.9f,0.2f},{0,0,0}},
    };
    int n=5; cudaMemcpyToSymbol(d_s,hs,sizeof(hs)); cudaMemcpyToSymbol(d_n,&n,4);
    const int W=480,H=270;
    uint8_t* d; cudaMalloc(&d,W*H*3);
    dim3 b(16,16),g((W+15)/16,(H+15)/16);
    // 教学：用较小 spp 演示；注释可改为 1024
    struct {const char* name; int mode; int spp;} jobs[]={
        {"ao",0,32}, {"gi1",1,64}, {"path",2,128},
    };
    for(auto j: jobs){
        auto t0=std::chrono::high_resolution_clock::now();
        k_render<<<g,b>>>(d,W,H,j.spp,j.mode);
        cudaDeviceSynchronize();
        auto t1=std::chrono::high_resolution_clock::now();
        std::vector<uint8_t> img(W*H*3);
        cudaMemcpy(img.data(),d,img.size(),cudaMemcpyDeviceToHost);
        write_ppm(std::string("output/")+j.name+".ppm", img, W, H);
        printf("%s spp=%d: %.1f ms\n", j.name, j.spp, std::chrono::duration<double,std::milli>(t1-t0).count());
    }
    printf("提示: 将 path spp 提到 1024 可接近照片级（更慢）。\n");
    printf("✓ AO + GI + 路径追踪完成。\n");
    return 0;
}
