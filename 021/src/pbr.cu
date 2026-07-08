// 021 | PBR：Cook-Torrance + 预计算 BRDF LUT + 简易 IBL + Bloom + tonemap
#include "rt_math.cuh"
#include <cstdio>
#include <chrono>
#include <vector>

__device__ float D_GGX(float NoH, float a){ float a2=a*a; float d=NoH*NoH*(a2-1)+1; return a2/(3.14159265f*d*d+1e-7f); }
__device__ float G_Schlick(float NoV, float NoL, float k){
    auto g=[&](float n){return n/(n*(1-k)+k+1e-7f);}; return g(NoV)*g(NoL);
}
__device__ Vec3 F_Schlick(float cosT, Vec3 F0) {
    float m=1-cosT; float m2=m*m; float m5=m2*m2*m;
    return F0 + (Vec3{1,1,1}-F0)*m5;
}

// 预积分 BRDF LUT：scale, bias for split-sum
__global__ void k_brdf_lut(float* lut, int N){
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=N||y>=N)return;
    float NoV = (x+0.5f)/N;
    float roughness = (y+0.5f)/N;
    // 简化 Monte Carlo
    float A=0,B=0; const int S=64;
    for(int i=0;i<S;++i){
        float u1=(i+0.5f)/S, u2=fmodf(i*0.754877f,1.f);
        float phi=6.2831853f*u1; float cosT=sqrtf((1-u2)/(1+(roughness*roughness-1)*u2)); float sinT=sqrtf(1-cosT*cosT);
        Vec3 H={sinT*cosf(phi), sinT*sinf(phi), cosT};
        Vec3 V={sqrtf(1-NoV*NoV),0,NoV};
        Vec3 L=normalize(H*2.f*dot(V,H) - V);
        float NoL=fmaxf(0.f,L.z), NoH=fmaxf(0.f,H.z), VoH=fmaxf(0.f,dot(V,H));
        if(NoL>0){
            float G=G_Schlick(NoV,NoL, roughness*roughness*0.5f);
            float G_Vis = G*VoH/(NoH*NoV+1e-5f);
            float Fc=powf(1-VoH,5.f);
            A+=(1-Fc)*G_Vis; B+=Fc*G_Vis;
        }
    }
    lut[(y*N+x)*2]=A/S; lut[(y*N+x)*2+1]=B/S;
}

__device__ Vec3 shade_pbr(Vec3 N, Vec3 V, Vec3 L, Vec3 albedo, float metallic, float roughness, Vec3 light_col) {
    Vec3 H=normalize(V+L);
    float NoV=fmaxf(0.001f,dot(N,V)), NoL=fmaxf(0.f,dot(N,L)), NoH=fmaxf(0.f,dot(N,H)), VoH=fmaxf(0.f,dot(V,H));
    float a=fmaxf(0.001f,roughness*roughness);
    Vec3 F0 = Vec3{0.04f,0.04f,0.04f}*(1-metallic) + albedo*metallic;
    float D=D_GGX(NoH,a);
    float G=G_Schlick(NoV,NoL, a*0.5f);
    Vec3 F=F_Schlick(VoH,F0);
    Vec3 spec = F * (D*G / fmaxf(1e-4f, 4*NoV*NoL));
    Vec3 kd = (Vec3{1,1,1}-F)*(1-metallic);
    Vec3 diff = kd * albedo * (1.f/3.14159265f);
    return (diff+spec)*light_col*NoL;
}

// 简易环境辐照：常数 + 方向
__device__ Vec3 ibl_diffuse(Vec3 N, Vec3 albedo, float metallic) {
    Vec3 irr = Vec3{0.3f,0.35f,0.45f} + N*Vec3{0.1f,0.15f,0.2f};
    // 只对正分量
    irr = Vec3{fabsf(irr.x),fabsf(irr.y),fabsf(irr.z)};
    return albedo*(1-metallic)*irr;
}

__global__ void k_render(uint8_t* out, float* hdr, const float* lut, int W, int H, int lutN){
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=W||y>=H)return;
    float u=(2.f*(x+0.5f)/W-1.f)*(float)W/H, v=1.f-2.f*(y+0.5f)/H;
    Vec3 ro{0,0.3f,2.2f}, rd=normalize(Vec3{u,v,-1.5f});
    // 球体材质网格：粗糙度 x 金属度
    float best=1e20f; Vec3 N,alb; float met=0,rough=0.5f; bool hit=false;
    for(int j=0;j<3;++j)for(int i=0;i<3;++i){
        Vec3 c{(i-1)*0.7f, (j-1)*0.7f, -1.2f}; float r=0.28f;
        Vec3 oc=ro-c; float a=dot(rd,rd),b=2*dot(oc,rd),cc=dot(oc,oc)-r*r;
        float d=b*b-4*a*cc; if(d<0)continue; float t=(-b-sqrtf(d))/(2*a);
        if(t>0.01f&&t<best){ best=t; N=normalize(ro+rd*t-c); alb=Vec3{0.9f,0.6f,0.2f}; met=i/2.f; rough=fmaxf(0.05f,j/2.f); hit=true; }
    }
    if(fabsf(rd.y)>1e-6f){ float t=(-0.8f-ro.y)/rd.y; if(t>0.01f&&t<best){best=t;N=Vec3{0,1,0};alb=Vec3{0.5f,0.5f,0.5f};met=0;rough=0.9f;hit=true;} }
    Vec3 col{0.02f,0.02f,0.03f};
    if(hit){
        Vec3 p=ro+rd*best; Vec3 V=normalize(ro-p);
        Vec3 L=normalize(Vec3{2,3,1}-p);
        col = shade_pbr(N,V,L,alb,met,rough,Vec3{3,3,3});
        col = col + ibl_diffuse(N,alb,met);
        // split-sum 用 LUT
        float NoV=fmaxf(0.001f,dot(N,V));
        int lx=min(lutN-1,(int)(NoV*lutN)), ly=min(lutN-1,(int)(rough*lutN));
        float A=lut[(ly*lutN+lx)*2], B=lut[(ly*lutN+lx)*2+1];
        Vec3 F0=Vec3{0.04f,0.04f,0.04f}*(1-met)+alb*met;
        Vec3 envSpec = Vec3{0.4f,0.45f,0.55f}; // 预过滤环境近似
        col = col + envSpec*(F0*A + Vec3{B,B,B});
    }
    // Bloom 阈值提取稍后 host
    int i=y*W+x;
    hdr[i*3]=col.x; hdr[i*3+1]=col.y; hdr[i*3+2]=col.z;
}

__global__ void k_bloom_tonemap(const float* hdr, uint8_t* out, int W, int H){
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=W||y>=H)return;
    // 简易邻域 bloom
    float bloom[3]={0,0,0}; int cnt=0;
    for(int dy=-2;dy<=2;++dy)for(int dx=-2;dx<=2;++dx){
        int ix=min(W-1,max(0,x+dx)), iy=min(H-1,max(0,y+dy));
        float r=hdr[(iy*W+ix)*3], g=hdr[(iy*W+ix)*3+1], b=hdr[(iy*W+ix)*3+2];
        float Y=0.2126f*r+0.7152f*g+0.0722f*b;
        if(Y>1.0f){ bloom[0]+=r; bloom[1]+=g; bloom[2]+=b; cnt++; }
    }
    if(cnt){ bloom[0]/=cnt; bloom[1]/=cnt; bloom[2]/=cnt; }
    float r=hdr[(y*W+x)*3]+bloom[0]*0.3f;
    float g=hdr[(y*W+x)*3+1]+bloom[1]*0.3f;
    float b=hdr[(y*W+x)*3+2]+bloom[2]*0.3f;
    // ACES 近似
    auto aces=[](float v){ return fminf(1.f, fmaxf(0.f, (v*(2.51f*v+0.03f))/(v*(2.43f*v+0.59f)+0.14f))); };
    r=sqrtf(aces(r)); g=sqrtf(aces(g)); b=sqrtf(aces(b)); // gamma
    int i=(y*W+x)*3; out[i]=(uint8_t)(r*255); out[i+1]=(uint8_t)(g*255); out[i+2]=(uint8_t)(b*255);
}

int main(){
    printf("============================================================\n");
    printf("021 | PBR Cook-Torrance + IBL LUT + Bloom\n");
    printf("============================================================\n");
    const int lutN=64;
    float* d_lut; cudaMalloc(&d_lut, lutN*lutN*2*sizeof(float));
    dim3 lb(16,16), lg((lutN+15)/16,(lutN+15)/16);
    k_brdf_lut<<<lg,lb>>>(d_lut, lutN); cudaDeviceSynchronize();
    const int W=640,H=360;
    float* d_hdr; uint8_t* d_out;
    cudaMalloc(&d_hdr,W*H*3*sizeof(float)); cudaMalloc(&d_out,W*H*3);
    dim3 b(16,16), g((W+15)/16,(H+15)/16);
    k_render<<<g,b>>>(d_out,d_hdr,d_lut,W,H,lutN);
    k_bloom_tonemap<<<g,b>>>(d_hdr,d_out,W,H);
    cudaDeviceSynchronize();
    std::vector<uint8_t> img(W*H*3); std::vector<float> lut(lutN*lutN*2);
    cudaMemcpy(img.data(),d_out,img.size(),cudaMemcpyDeviceToHost);
    cudaMemcpy(lut.data(),d_lut,lut.size()*4,cudaMemcpyDeviceToHost);
    write_ppm("output/pbr.ppm", img, W, H);
    // 保存 LUT 可视化
    std::vector<uint8_t> lutimg(lutN*lutN*3);
    for(int i=0;i<lutN*lutN;++i){ lutimg[i*3]=(uint8_t)(fminf(1.f,lut[i*2])*255); lutimg[i*3+1]=(uint8_t)(fminf(1.f,lut[i*2+1])*255); lutimg[i*3+2]=0; }
    write_ppm("output/brdf_lut.ppm", lutimg, lutN, lutN);
    printf("金属度×粗糙度 3x3 球；工作流: metallic-roughness\n");
    printf("✓ PBR 管线完成。\n");
    return 0;
}
