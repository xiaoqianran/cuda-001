// 015 | 光追优化：分 tile 流、材质分组避免 divergence、常量内存场景
#include "rt_math.cuh"
#include <cstdio>
#include <chrono>
#include <vector>

struct Sphere { Vec3 c; float r; int mat_id; };
struct Mat { Vec3 albedo; int type; }; // 0 lambert 1 metal

__constant__ Sphere d_sph[16];
__constant__ Mat d_mat[8];
__constant__ int d_ns, d_nm;

__device__ bool hit(Ray r, float& t, Vec3& n, int& mid){
    bool ok=false; t=1e20f;
    for(int i=0;i<d_ns;++i){
        Vec3 oc=r.o-d_sph[i].c; float a=dot(r.d,r.d), hb=dot(oc,r.d), c=dot(oc,oc)-d_sph[i].r*d_sph[i].r;
        float disc=hb*hb-a*c; if(disc<0)continue;
        float sq=sqrtf(disc); float root=(-hb-sq)/a;
        if(root<1e-3f||root>t){root=(-hb+sq)/a; if(root<1e-3f||root>t)continue;}
        t=root; n=normalize(r.o+r.d*t-d_sph[i].c); mid=d_sph[i].mat_id; ok=true;
    }
    return ok;
}

// 基线：混合材质可能 warp divergence
__global__ void k_baseline(uint8_t* out, int W, int H){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=W||y>=H)return;
    float aspect=(float)W/H;
    float u=(2.f*(x+.5f)/W-1.f)*aspect, v=1.f-2.f*(y+.5f)/H;
    Ray r{Vec3{0,0,2}, normalize(Vec3{u,v,-1.5f})};
    Vec3 col{0.1f,0.1f,0.15f};
    float t; Vec3 n; int mid;
    if(hit(r,t,n,mid)){
        Mat m=d_mat[mid];
        if(m.type==0){
            float ndl=fmaxf(0.f,dot(n,normalize(Vec3{1,1,1})));
            col=m.albedo*(0.2f+0.8f*ndl);
        } else {
            // metal 二次反射（故意制造 divergence）
            Ray r2{r.o+r.d*t+n*1e-3f, reflect(r.d,n)};
            float t2; Vec3 n2; int m2;
            if(hit(r2,t2,n2,m2)) col=d_mat[m2].albedo*m.albedo;
            else col=m.albedo*Vec3{0.5f,0.6f,0.8f};
        }
    }
    int i=(y*W+x)*3; out[i]=(uint8_t)(fminf(1.f,col.x)*255); out[i+1]=(uint8_t)(fminf(1.f,col.y)*255); out[i+2]=(uint8_t)(fminf(1.f,col.z)*255);
}

// 优化：两 pass 材质分组 —— 先写 mat id，再分别处理
__global__ void k_gbuffer(int* mat_buf, float* t_buf, float* n_buf, int W, int H){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=W||y>=H)return;
    int idx=y*W+x;
    float aspect=(float)W/H;
    float u=(2.f*(x+.5f)/W-1.f)*aspect, v=1.f-2.f*(y+.5f)/H;
    Ray r{Vec3{0,0,2}, normalize(Vec3{u,v,-1.5f})};
    float t; Vec3 n; int mid;
    if(hit(r,t,n,mid)){ mat_buf[idx]=mid; t_buf[idx]=t; n_buf[idx*3]=n.x; n_buf[idx*3+1]=n.y; n_buf[idx*3+2]=n.z; }
    else mat_buf[idx]=-1;
}

__global__ void k_shade_grouped(const int* mat_buf, const float* t_buf, const float* n_buf, uint8_t* out, int W, int H, int only_type){
    // only_type: 只处理该材质类型，减少 warp 内分支
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=W||y>=H)return;
    int idx=y*W+x; int mid=mat_buf[idx];
    if(mid<0){
        if(only_type==0){ int i=idx*3; out[i]=25; out[i+1]=25; out[i+2]=40; }
        return;
    }
    Mat m=d_mat[mid];
    if(m.type!=only_type) return;
    Vec3 n{n_buf[idx*3], n_buf[idx*3+1], n_buf[idx*3+2]};
    Vec3 col;
    if(m.type==0){
        float ndl=fmaxf(0.f,dot(n,normalize(Vec3{1,1,1})));
        col=m.albedo*(0.2f+0.8f*ndl);
    } else {
        float aspect=(float)W/H;
        float u=(2.f*(x+.5f)/W-1.f)*aspect, v=1.f-2.f*(y+.5f)/H;
        Ray r{Vec3{0,0,2}, normalize(Vec3{u,v,-1.5f})};
        Vec3 p=r.o+r.d*t_buf[idx];
        Ray r2{p+n*1e-3f, reflect(r.d,n)};
        float t2; Vec3 n2; int m2;
        if(hit(r2,t2,n2,m2)) col=d_mat[m2].albedo*m.albedo;
        else col=m.albedo*Vec3{0.5f,0.6f,0.8f};
    }
    int i=idx*3;
    out[i]=(uint8_t)(fminf(1.f,col.x)*255);
    out[i+1]=(uint8_t)(fminf(1.f,col.y)*255);
    out[i+2]=(uint8_t)(fminf(1.f,col.z)*255);
}

int main(){
    printf("============================================================\n");
    printf("015 | 光追优化 (分组着色 + streams)\n");
    printf("============================================================\n");
    Sphere hs[8]; Mat ms[4];
    for(int i=0;i<8;++i){
        hs[i].c=Vec3{(i%4-1.5f)*0.7f, (i/4-0.5f)*0.7f, -1.5f};
        hs[i].r=0.3f; hs[i].mat_id=i%2;
    }
    ms[0]={{0.8f,0.3f,0.2f},0}; ms[1]={{0.9f,0.9f,0.95f},1};
    int ns=8,nm=2;
    cudaMemcpyToSymbol(d_sph,hs,sizeof(hs)); cudaMemcpyToSymbol(d_mat,ms,sizeof(ms));
    cudaMemcpyToSymbol(d_ns,&ns,4); cudaMemcpyToSymbol(d_nm,&nm,4);
    const int W=800,H=600;
    uint8_t *d1,*d2; cudaMalloc(&d1,W*H*3); cudaMalloc(&d2,W*H*3);
    int* d_m; float* d_t; float* d_n;
    cudaMalloc(&d_m,W*H*4); cudaMalloc(&d_t,W*H*4); cudaMalloc(&d_n,W*H*12);
    dim3 b(16,16), g((W+15)/16,(H+15)/16);

    auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<50;++i) k_baseline<<<g,b>>>(d1,W,H);
    cudaDeviceSynchronize();
    auto t1=std::chrono::high_resolution_clock::now();
    double ms_base=std::chrono::duration<double,std::milli>(t1-t0).count()/50;

    // 双流：可并行准备（演示 streams）
    cudaStream_t s0,s1; cudaStreamCreate(&s0); cudaStreamCreate(&s1);
    t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<50;++i){
        k_gbuffer<<<g,b,0,s0>>>(d_m,d_t,d_n,W,H);
        cudaStreamSynchronize(s0);
        k_shade_grouped<<<g,b,0,s0>>>(d_m,d_t,d_n,d2,W,H,0);
        k_shade_grouped<<<g,b,0,s1>>>(d_m,d_t,d_n,d2,W,H,1);
        cudaStreamSynchronize(s0); cudaStreamSynchronize(s1);
    }
    t1=std::chrono::high_resolution_clock::now();
    double ms_opt=std::chrono::duration<double,std::milli>(t1-t0).count()/50;
    printf("基线: %.3f ms | 材质分组+streams: %.3f ms | 比: %.2fx\n", ms_base, ms_opt, ms_base/ms_opt);
    std::vector<uint8_t> img(W*H*3);
    cudaMemcpy(img.data(),d2,img.size(),cudaMemcpyDeviceToHost);
    write_ppm("output/opt.ppm", img, W, H);
    printf("✓ 优化对比报告完成。\n");
    cudaStreamDestroy(s0); cudaStreamDestroy(s1);
    return 0;
}
