// 017 | 小型离线渲染器：场景驱动 + 多材质路径追踪 + 简易自适应采样
// JSON 解析用极简手写（避免依赖）；输出 PPM + 浮点 HDR (.pfm 风格头)
#include "rt_math.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>
#include <cmath>

enum CamType { PERSPECTIVE=0, ORTHO=1, FISHEYE=2 };
enum MType { LAMBERT=0, METAL=1, DIELECTRIC=2, EMISSIVE=3 };

struct Material { MType type; Vec3 albedo; float roughness, ior; Vec3 emission; };
struct Sphere { Vec3 c; float r; int mat; };
struct Camera {
    CamType type; Vec3 pos, lookat, up; float fov;
};

// 极简：内置 demo 场景（也可从 JSON 关键字扫描）
static bool load_scene_simple(const char* path, Camera& cam, std::vector<Sphere>& sph,
                              std::vector<Material>& mats, int& W, int& H, int& spp) {
    // 默认
    cam = {PERSPECTIVE, {0,0.5f,2.5f}, {0,0,-1}, {0,1,0}, 45.f};
    W=640; H=360; spp=48;
    mats = {
        {LAMBERT, {0.8f,0.2f,0.2f}, 0,1,{0,0,0}},
        {METAL, {0.9f,0.9f,0.95f}, 0.05f,1,{0,0,0}},
        {DIELECTRIC, {1,1,1}, 0,1.5f,{0,0,0}},
        {EMISSIVE, {1,1,1}, 0,1,{10,9,8}},
    };
    sph = {
        {{0,0,-1},0.5f,0}, {{-1.1f,0,-1.2f},0.45f,1}, {{1.0f,0,-1.3f},0.4f,2}, {{0,1.4f,-0.8f},0.25f,3},
    };
    // 尝试读 JSON 里的 resolution / spp
    std::ifstream in(path);
    if(!in) return false;
    std::string s((std::istreambuf_iterator<char>(in)), {});
    auto find_int=[&](const char* key, int def){
        auto p=s.find(key); if(p==std::string::npos) return def;
        p=s.find(':',p); if(p==std::string::npos) return def;
        return atoi(s.c_str()+p+1);
    };
    spp = find_int("\"spp\"", spp);
    // resolution [w,h]
    auto pr=s.find("\"resolution\"");
    if(pr!=std::string::npos){
        auto lb=s.find('[',pr); auto com=s.find(',',lb); auto rb=s.find(']',com);
        if(lb!=std::string::npos){ W=atoi(s.c_str()+lb+1); H=atoi(s.c_str()+com+1); }
    }
    // camera type
    if(s.find("\"ortho\"")!=std::string::npos) cam.type=ORTHO;
    if(s.find("\"fisheye\"")!=std::string::npos) cam.type=FISHEYE;
    printf("场景: %s  res=%dx%d spp=%d cam=%d spheres=%zu\n", path, W,H,spp,(int)cam.type,sph.size());
    return true;
}

struct GPUScene { Sphere s[16]; Material m[8]; int ns, nm; Camera cam; };
__constant__ GPUScene d_sc;

__device__ float rnd(unsigned& s){ s=s*1664525u+1013904223u; return (s&0xFFFFFF)/16777216.f; }
__device__ Vec3 rand_unit(unsigned& s) {
    float z=rnd(s)*2-1,a=rnd(s)*6.2831853f; float r=sqrtf(fmaxf(0.f,1-z*z)); return {r*cosf(a),r*sinf(a),z};
}

__device__ Ray gen_ray(Camera cam, float u, float v, float aspect){
    // u,v in [-1,1]
    Vec3 w=normalize(cam.pos-cam.lookat);
    Vec3 uu=normalize(cross(cam.up,w));
    Vec3 vv=cross(w,uu);
    if(cam.type==ORTHO){
        Vec3 o=cam.pos + uu*(u*aspect) + vv*v;
        return {o, normalize(-w)};
    }
    if(cam.type==FISHEYE){
        float r=sqrtf(u*u+v*v); if(r>1) r=1;
        float theta=r*cam.fov*0.5f*3.14159265f/180.f;
        float phi=atan2f(v,u);
        Vec3 d = uu*(sinf(theta)*cosf(phi)) + vv*(sinf(theta)*sinf(phi)) + (-w)*cosf(theta);
        return {cam.pos, normalize(d)};
    }
    // perspective
    float scale = tanf(cam.fov*0.5f*3.14159265f/180.f);
    Vec3 d = normalize(uu*(u*aspect*scale) + vv*(v*scale) - w);
    return {cam.pos, d};
}

__device__ bool world_hit(Ray r, float& t, Vec3& n, Material& mat){
    bool ok=false; t=1e20f;
    for(int i=0;i<d_sc.ns;++i){
        Vec3 oc=r.o-d_sc.s[i].c; float a=dot(r.d,r.d),hb=dot(oc,r.d),c=dot(oc,oc)-d_sc.s[i].r*d_sc.s[i].r;
        float disc=hb*hb-a*c; if(disc<0)continue;
        float sq=sqrtf(disc); float root=(-hb-sq)/a;
        if(root<1e-3f||root>t){root=(-hb+sq)/a; if(root<1e-3f||root>t)continue;}
        t=root; n=normalize(r.o+r.d*t-d_sc.s[i].c); mat=d_sc.m[d_sc.s[i].mat]; ok=true;
    }
    if(fabsf(r.d.y)>1e-6f){
        float tp=(-0.5f-r.o.y)/r.d.y;
        if(tp>1e-3f&&tp<t){ t=tp; n=Vec3{0,1,0}; mat={LAMBERT,{0.75f,0.75f,0.75f},0,1,{0,0,0}}; ok=true; }
    }
    return ok;
}

__device__ Vec3 path(Ray r, unsigned& s) {
    Vec3 col{0,0,0}, thr{1,1,1};
    for(int d=0;d<6;++d){
        float t; Vec3 n; Material m;
        if(!world_hit(r,t,n,m)){ col=col+thr*Vec3{0.4f,0.5f,0.7f}*0.4f; break; }
        Vec3 p=r.o+r.d*t;
        col = col + thr*m.emission;
        if(m.type==EMISSIVE) break;
        Vec3 atten; Ray sc;
        if(m.type==LAMBERT){
            Vec3 dir=normalize(n+rand_unit(s)); sc={p+n*1e-3f,dir}; atten=m.albedo;
        } else if(m.type==METAL){
            Vec3 dir=normalize(reflect(r.d,n)+rand_unit(s)*m.roughness);
            sc={p+n*1e-3f,dir}; atten=m.albedo; if(dot(dir,n)<=0)break;
        } else {
            // dielectric 简化：只反射
            Vec3 dir=reflect(normalize(r.d),n); sc={p+n*1e-3f,dir}; atten=Vec3{1,1,1};
        }
        thr=thr*atten; r=sc;
        if(d>2){ float pc=fminf(0.95f,fmaxf(thr.x,fmaxf(thr.y,thr.z))); if(rnd(s)>pc)break; thr=thr*(1.f/pc); }
    }
    return col;
}

// 自适应：先低 spp，方差高再补采样
__global__ void k_render(float* hdr, uint8_t* ldr, int W, int H, int spp_base, int spp_extra){
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=W||y>=H)return;
    unsigned s=x*1973u+y*9176u+1u;
    float aspect=(float)W/H;
    Vec3 acc{0,0,0}; float m2=0; // 简单亮度二阶矩
    int spp=spp_base;
    for(int i=0;i<spp;++i){
        float u=2.f*(x+rnd(s))/W-1.f, v=1.f-2.f*(y+rnd(s))/H;
        Ray r=gen_ray(d_sc.cam,u,v,aspect);
        Vec3 c=path(r,s); acc=acc+c;
        float Y=0.2126f*c.x+0.7152f*c.y+0.0722f*c.z; m2+=Y*Y;
    }
    Vec3 mean=acc*(1.f/spp);
    float meanY=0.2126f*mean.x+0.7152f*mean.y+0.0722f*mean.z;
    float var = m2/spp - meanY*meanY;
    // 方差大则额外采样
    if(var > 0.01f){
        for(int i=0;i<spp_extra;++i){
            float u=2.f*(x+rnd(s))/W-1.f, v=1.f-2.f*(y+rnd(s))/H;
            acc = acc + path(gen_ray(d_sc.cam,u,v,aspect), s);
        }
        spp += spp_extra;
        mean = acc*(1.f/spp);
    }
    // 简易 NLM 邻域不在此做；写 HDR + LDR gamma
    int i=y*W+x;
    hdr[i*3]=mean.x; hdr[i*3+1]=mean.y; hdr[i*3+2]=mean.z;
    auto g=[](float v){return sqrtf(fminf(1.f,fmaxf(0.f,v)));};
    ldr[i*3]=(uint8_t)(g(mean.x)*255);
    ldr[i*3+1]=(uint8_t)(g(mean.y)*255);
    ldr[i*3+2]=(uint8_t)(g(mean.z)*255);
}

static void write_pfm(const char* path, const float* hdr, int W, int H){
    // HDR 浮点输出（PBRT 风格可换 OpenEXR；此处 PFM）
    FILE* f=fopen(path,"wb");
    fprintf(f,"PF\n%d %d\n-1.0\n",W,H);
    for(int y=H-1;y>=0;--y) fwrite(hdr+y*W*3, sizeof(float), W*3, f);
    fclose(f);
}

int main(int argc, char** argv){
    printf("============================================================\n");
    printf("017 | 小型离线渲染器 (JSON 场景驱动)\n");
    printf("============================================================\n");
    const char* scene = argc>1 ? argv[1] : "scenes/demo.json";
    Camera cam; std::vector<Sphere> sph; std::vector<Material> mats; int W,H,spp;
    if(!load_scene_simple(scene, cam, sph, mats, W, H, spp)){
        fprintf(stderr,"无法加载场景，使用内置默认\n");
        load_scene_simple("scenes/demo.json", cam, sph, mats, W, H, spp);
    }
    GPUScene gs{}; gs.cam=cam; gs.ns=(int)sph.size(); gs.nm=(int)mats.size();
    for(int i=0;i<gs.ns;++i) gs.s[i]=sph[i];
    for(int i=0;i<gs.nm;++i) gs.m[i]=mats[i];
    cudaMemcpyToSymbol(d_sc,&gs,sizeof(gs));
    float* d_hdr; uint8_t* d_ldr;
    cudaMalloc(&d_hdr,W*H*3*sizeof(float)); cudaMalloc(&d_ldr,W*H*3);
    dim3 b(16,16), g((W+15)/16,(H+15)/16);
    auto t0=std::chrono::high_resolution_clock::now();
    k_render<<<g,b>>>(d_hdr,d_ldr,W,H, spp/2, spp/2);
    cudaDeviceSynchronize();
    auto t1=std::chrono::high_resolution_clock::now();
    printf("渲染完成: %.1f ms\n", std::chrono::duration<double,std::milli>(t1-t0).count());
    std::vector<uint8_t> ldr(W*H*3); std::vector<float> hdr(W*H*3);
    cudaMemcpy(ldr.data(),d_ldr,ldr.size(),cudaMemcpyDeviceToHost);
    cudaMemcpy(hdr.data(),d_hdr,hdr.size()*4,cudaMemcpyDeviceToHost);
    write_ppm("output/beauty.ppm", ldr, W, H);
    write_pfm("output/beauty.pfm", hdr.data(), W, H);
    printf("输出: output/beauty.ppm + beauty.pfm (HDR)\n");
    printf("✓ 小型 PBRT 管线骨架完成。\n");
    return 0;
}
