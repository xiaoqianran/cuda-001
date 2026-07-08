// 013 | 金属 / 电介质 / 发光 / Cook-Torrance 微表面
#include "rt_math.cuh"
#include <cstdio>
#include <chrono>

enum MatType { LAMBERT=0, METAL=1, DIELECTRIC=2, EMISSIVE=3, COOK_TORRANCE=4 };
struct Material {
    MatType type;
    Vec3 albedo;
    float roughness; // metal fuzzy / CT
    float ior;       // 电介质折射率
    Vec3 emission;
    float metallic;
};

struct Sphere { Vec3 c; float r; Material m; };

__device__ float rand01(unsigned& s) {
    s = s * 1664525u + 1013904223u;
    return (s & 0xFFFFFF) / 16777216.f;
}
__device__ Vec3 rand_unit(unsigned& s) {
    float z = rand01(s)*2-1, a = rand01(s)*6.2831853f;
    float r = sqrtf(fmaxf(0.f,1-z*z));
    return {r*cosf(a), r*sinf(a), z};
}

__device__ bool hit_sph(Sphere s, Ray r, float tmin, float tmax, float& t, Vec3& n, Material& m) {
    Vec3 oc=r.o-s.c; float a=dot(r.d,r.d), half_b=dot(oc,r.d), c=dot(oc,oc)-s.r*s.r;
    float disc=half_b*half_b-a*c; if(disc<0)return false;
    float sq=sqrtf(disc); float root=(-half_b-sq)/a;
    if(root<tmin||root>tmax){root=(-half_b+sq)/a; if(root<tmin||root>tmax)return false;}
    t=root; n=normalize(r.o+r.d*t - s.c); m=s.m; return true;
}

// Schlick Fresnel
__device__ float fresnel(float cos_i, float ior) {
    float r0 = (1-ior)/(1+ior); r0*=r0;
    float m = 1-cos_i; return r0+(1-r0)*m*m*m*m*m;
}

// GGX NDF
__device__ float D_GGX(float NdotH, float a) {
    float a2=a*a; float d=NdotH*NdotH*(a2-1)+1;
    return a2 / (3.14159265f * d * d);
}
__device__ float G_Smith(float NdotV, float NdotL, float a) {
    auto G1=[&](float nd){ float k=a*0.5f; return nd/(nd*(1-k)+k+1e-5f); };
    return G1(NdotV)*G1(NdotL);
}

struct Scene { Sphere s[6]; int n; };

__constant__ Scene d_scene;

__device__ bool closest(Ray r, float& t, Vec3& n, Material& m) {
    bool ok=false; t=1e20f;
    for(int i=0;i<d_scene.n;++i){
        float tt; Vec3 nn; Material mm;
        if(hit_sph(d_scene.s[i], r, 0.001f, t, tt, nn, mm)){ t=tt; n=nn; m=mm; ok=true; }
    }
    // 地面平面 y=-0.5
    if(fabsf(r.d.y)>1e-6f){
        float tp = (-0.5f - r.o.y)/r.d.y;
        if(tp>0.001f && tp<t){
            t=tp; n=Vec3{0,1,0};
            // 棋盘 Lambert
            Vec3 p=r.o+r.d*t;
            int cx=(int)floorf(p.x), cz=(int)floorf(p.z);
            float c = ((cx+cz)&1) ? 0.8f : 0.3f;
            m = {LAMBERT, {c,c,c}, 0,1,{0,0,0},0};
            ok=true;
        }
    }
    return ok;
}

// 材质散射：GPU 上用枚举分发（“虚函数”替代）
__device__ bool scatter(Material m, Ray r_in, Vec3 p, Vec3 n, unsigned& rng,
                        Ray& scattered, Vec3& atten, Vec3& emitted) {
    emitted = m.emission;
    if (m.type == EMISSIVE) { atten=Vec3{0,0,0}; return false; }
    if (m.type == LAMBERT) {
        Vec3 dir = normalize(n + rand_unit(rng));
        if (dot(dir,n)<0) dir = n;
        scattered = {p + n*1e-3f, dir};
        atten = m.albedo;
        return true;
    }
    if (m.type == METAL) {
        Vec3 refl = reflect(normalize(r_in.d), n);
        refl = normalize(refl + rand_unit(rng)*m.roughness); // fuzzy
        scattered = {p + n*1e-3f, refl};
        atten = m.albedo;
        return dot(refl, n) > 0;
    }
    if (m.type == DIELECTRIC) {
        // Snell 折射 + Fresnel
        float ior = m.ior;
        Vec3 unit = normalize(r_in.d);
        float cos_theta = fminf(dot(-unit, n), 1.f);
        float sin_theta = sqrtf(1 - cos_theta*cos_theta);
        bool front = dot(r_in.d, n) < 0;
        float etai_over_etat = front ? (1.f/ior) : ior;
        Vec3 outward = front ? n : n*(-1.f);
        bool cannot = etai_over_etat * sin_theta > 1.f;
        Vec3 dir;
        if (cannot || fresnel(cos_theta, ior) > rand01(rng))
            dir = reflect(unit, outward);
        else {
            // 折射
            Vec3 r_out_perp = (unit + outward*cos_theta) * etai_over_etat;
            Vec3 r_out_par = outward * (-sqrtf(fabsf(1 - dot(r_out_perp,r_out_perp))));
            dir = r_out_perp + r_out_par;
        }
        scattered = {p + dir*1e-3f, normalize(dir)};
        atten = Vec3{1,1,1};
        return true;
    }
    if (m.type == COOK_TORRANCE) {
        // 简化：重要性采样近似 → 一半漫反射一半镜面
        Vec3 V = normalize(-r_in.d);
        Vec3 L = normalize(n + rand_unit(rng));
        Vec3 H = normalize(V+L);
        float NdotL=fmaxf(0.f,dot(n,L)), NdotV=fmaxf(0.f,dot(n,V)), NdotH=fmaxf(0.f,dot(n,H)), VdotH=fmaxf(0.f,dot(V,H));
        float a = fmaxf(0.02f, m.roughness*m.roughness);
        float D = D_GGX(NdotH, a);
        float G = G_Smith(NdotV, NdotL, a);
        float F = fresnel(VdotH, 0.04f + 0.96f*m.metallic);
        Vec3 spec = m.albedo * (D*G*F / fmaxf(1e-4f, 4*NdotV*NdotL));
        Vec3 diff = m.albedo * (1.f - m.metallic) * (1.f/3.14159265f);
        atten = (diff + spec) * NdotL * 3.14159265f;
        scattered = {p+n*1e-3f, L};
        return NdotL > 0;
    }
    return false;
}

__device__ Vec3 path_trace(Ray r, unsigned rng) {
    Vec3 col{0,0,0}, thr{1,1,1};
    for (int depth=0; depth<6; ++depth) {
        float t; Vec3 n; Material m;
        if (!closest(r, t, n, m)) {
            float tt=0.5f*(r.d.y+1);
            col = col + thr*(Vec3{1,1,1}*(1-tt)+Vec3{0.5f,0.7f,1}*tt)*0.3f;
            break;
        }
        Vec3 p = r.o + r.d * t;
        Ray sc; Vec3 att, em;
        if (!scatter(m, r, p, n, rng, sc, att, em)) {
            col = col + thr * em;
            break;
        }
        col = col + thr * em;
        thr = thr * att;
        r = sc;
        // 俄罗斯轮盘赌
        if (depth > 2) {
            float pcont = fminf(0.95f, fmaxf(thr.x, fmaxf(thr.y, thr.z)));
            if (rand01(rng) > pcont) break;
            thr = thr * (1.f/pcont);
        }
    }
    return col;
}

__global__ void k_render(uint8_t* out, int W, int H, int spp) {
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=W||y>=H) return;
    unsigned rng = x + y*W + 1;
    Vec3 acc{0,0,0};
    float aspect=(float)W/H;
    for(int s=0;s<spp;++s){
        float ju=rand01(rng), jv=rand01(rng);
        float u=(2.f*(x+ju)/W-1.f)*aspect;
        float v=1.f-2.f*(y+jv)/H;
        Ray r{Vec3{0,0.2f,1.8f}, normalize(Vec3{u,v-0.1f,-1.5f})};
        acc = acc + path_trace(r, rng);
    }
    acc = acc*(1.f/spp);
    auto tone=[](float v){ return sqrtf(fminf(1.f,v)); };
    int i=(y*W+x)*3;
    out[i]=(uint8_t)(tone(acc.x)*255);
    out[i+1]=(uint8_t)(tone(acc.y)*255);
    out[i+2]=(uint8_t)(tone(acc.z)*255);
}

int main() {
    printf("============================================================\n");
    printf("013 | 材质系统 (金属/玻璃/发光/Cook-Torrance)\n");
    printf("============================================================\n");
    Scene sc{};
    sc.n = 5;
    sc.s[0] = {{ -1.1f,0.0f,-1.5f}, 0.5f, {METAL, {0.9f,0.9f,0.95f}, 0.1f, 1, {0,0,0}, 1}};
    sc.s[1] = {{ 0.0f,0.0f,-1.8f}, 0.5f, {DIELECTRIC, {1,1,1}, 0, 1.5f, {0,0,0}, 0}};
    sc.s[2] = {{ 1.1f,0.0f,-1.5f}, 0.5f, {COOK_TORRANCE, {0.8f,0.4f,0.1f}, 0.3f, 1, {0,0,0}, 0.7f}};
    sc.s[3] = {{ 0.0f,1.2f,-1.2f}, 0.3f, {EMISSIVE, {1,1,1}, 0,1, {8,7,5}, 0}};
    sc.s[4] = {{ -0.3f,-0.2f,-2.5f}, 0.3f, {LAMBERT, {0.2f,0.6f,0.3f}, 0,1,{0,0,0},0}};
    cudaMemcpyToSymbol(d_scene, &sc, sizeof(sc));
    const int W=640,H=360, spp=32;
    uint8_t* d; cudaMalloc(&d,W*H*3);
    dim3 b(16,16), g((W+15)/16,(H+15)/16);
    auto t0=std::chrono::high_resolution_clock::now();
    k_render<<<g,b>>>(d,W,H,spp);
    cudaDeviceSynchronize();
    auto t1=std::chrono::high_resolution_clock::now();
    printf("渲染 %dx%d spp=%d: %.2f ms\n", W,H,spp, std::chrono::duration<double,std::milli>(t1-t0).count());
    std::vector<uint8_t> img(W*H*3);
    cudaMemcpy(img.data(),d,img.size(),cudaMemcpyDeviceToHost);
    write_ppm("output/materials.ppm", img, W, H);
    printf("✓ 材质多态分发 + 路径追踪完成。\n");
    return 0;
}
