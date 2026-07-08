// 020 | Shadow Mapping：深度 pass → 主 pass PCF；bias；简化 CSM 两级
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>
struct V3{float x,y,z;};
static V3 operator+(V3 a,V3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
static V3 operator-(V3 a,V3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
static V3 operator*(V3 a,float s){return {a.x*s,a.y*s,a.z*s};}
static float dot(V3 a,V3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static V3 normalize(V3 v){float L=sqrtf(dot(v,v));return L>0?v*(1/L):v;}

// 正交 shadow map：从光源方向投影
struct ShadowMap { int n; std::vector<float> depth; float extent; };

static void render_depth(ShadowMap& sm, V3 light_dir, const std::vector<V3>& spheres_c, const std::vector<float>& spheres_r){
    // 每个 shadow texel 沿光方向射线，记录最近深度
    sm.depth.assign(sm.n*sm.n, 1e30f);
    V3 up = fabsf(light_dir.y)<0.9f ? V3{0,1,0} : V3{1,0,0};
    V3 r = normalize({up.y*light_dir.z-up.z*light_dir.y, up.z*light_dir.x-up.x*light_dir.z, up.x*light_dir.y-up.y*light_dir.x});
    V3 u = {light_dir.y*r.z-light_dir.z*r.y, light_dir.z*r.x-light_dir.x*r.z, light_dir.x*r.y-light_dir.y*r.x};
    for(int y=0;y<sm.n;++y)for(int x=0;x<sm.n;++x){
        float fx=(x+0.5f)/sm.n*2-1, fy=(y+0.5f)/sm.n*2-1;
        V3 origin = r*(fx*sm.extent) + u*(fy*sm.extent) + light_dir*(-10.f);
        V3 dir = light_dir;
        float best=1e30f;
        for(size_t i=0;i<spheres_c.size();++i){
            // 射线-球
            V3 oc=origin-spheres_c[i]; float a=dot(dir,dir), b=2*dot(oc,dir), c=dot(oc,oc)-spheres_r[i]*spheres_r[i];
            float d=b*b-4*a*c; if(d<0)continue; float t=(-b-sqrtf(d))/(2*a); if(t>0&&t<best)best=t;
        }
        // 地面 y=0
        if(fabsf(dir.y)>1e-6f){ float t=-origin.y/dir.y; if(t>0&&t<best)best=t; }
        sm.depth[y*sm.n+x]=best;
    }
}

static float pcf(const ShadowMap& sm, V3 p, V3 light_dir, float bias){
    // 世界点投影到 shadow map
    V3 up = fabsf(light_dir.y)<0.9f ? V3{0,1,0} : V3{1,0,0};
    V3 r = normalize({up.y*light_dir.z-up.z*light_dir.y, up.z*light_dir.x-up.x*light_dir.z, up.x*light_dir.y-up.y*light_dir.x});
    V3 u = {light_dir.y*r.z-light_dir.z*r.y, light_dir.z*r.x-light_dir.x*r.z, light_dir.x*r.y-light_dir.y*r.x};
    V3 origin = light_dir*(-10.f);
    // 点在光空间的坐标
    V3 rel = p - origin;
    float depth = dot(rel, light_dir);
    float sx = dot(rel, r)/sm.extent; // -1..1
    float sy = dot(rel, u)/sm.extent;
    float u0=(sx*0.5f+0.5f)*sm.n, v0=(sy*0.5f+0.5f)*sm.n;
    // 3x3 PCF
    int vis=0, tot=0;
    for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
        int ix=(int)u0+dx, iy=(int)v0+dy;
        if(ix<0||iy<0||ix>=sm.n||iy>=sm.n) continue;
        tot++;
        if(depth - bias <= sm.depth[iy*sm.n+ix] + 1e-3f) vis++;
    }
    return tot? vis/(float)tot : 1.f;
}

// 简化 CSM：近 cascade + 远 cascade 两张图
int main(){
    printf("============================================================\n");
    printf("020 | Shadow Mapping + PCF + bias + CSM(2)\n");
    printf("============================================================\n");
    std::vector<V3> sc={{0,0.5f,-2},{-1,0.3f,-3},{1.2f,0.4f,-2.5f}};
    std::vector<float> sr={0.5f,0.3f,0.4f};
    V3 Ldir=normalize({0.4f,-1.f,0.3f});
    ShadowMap near_sm{512,{}, 4.f}, far_sm{512,{}, 12.f};
    render_depth(near_sm, Ldir, sc, sr);
    render_depth(far_sm, Ldir, sc, sr);
    const int W=640,H=480;
    std::vector<unsigned char> img(W*H*3);
    V3 eye{0,1.5f,2.5f};
    float bias=0.05f; // 解决 shadow acne
    for(int y=0;y<H;++y)for(int x=0;x<W;++x){
        float u=(x+0.5f)/W*2-1, v=1-(y+0.5f)/H*2;
        V3 rd=normalize({u*0.8f,v*0.6f,-1});
        // 求交
        float best=1e30f; V3 n{0,1,0}; V3 alb{0.7f,0.7f,0.7f}; bool hit=false;
        for(size_t i=0;i<sc.size();++i){
            V3 oc=eye-sc[i]; float a=dot(rd,rd),b=2*dot(oc,rd),c=dot(oc,oc)-sr[i]*sr[i];
            float d=b*b-4*a*c; if(d<0)continue; float t=(-b-sqrtf(d))/(2*a);
            if(t>0.01f&&t<best){best=t; n=normalize(eye+rd*t - sc[i]); alb={0.8f,0.3f,0.2f}; hit=true;}
        }
        if(fabsf(rd.y)>1e-6f){ float t=-eye.y/rd.y; if(t>0.01f&&t<best){best=t;n={0,1,0};alb={0.6f,0.65f,0.6f};hit=true;} }
        V3 col{0.1f,0.12f,0.18f};
        if(hit){
            V3 p=eye+rd*best;
            float dist=sqrtf(dot(p-eye,p-eye));
            // CSM 选择
            float shadow = dist < 5.f ? pcf(near_sm,p,Ldir,bias) : pcf(far_sm,p,Ldir,bias);
            float ndl=fmaxf(0.f,dot(n, Ldir*(-1.f)));
            col = alb*(0.15f + 0.85f*ndl*shadow);
        }
        int i=(y*W+x)*3;
        img[i]=(unsigned char)(std::min(1.f,col.x)*255);
        img[i+1]=(unsigned char)(std::min(1.f,col.y)*255);
        img[i+2]=(unsigned char)(std::min(1.f,col.z)*255);
    }
    FILE* f=fopen("output/shadow.ppm","wb"); fprintf(f,"P6\n%d %d\n255\n",W,H); fwrite(img.data(),1,img.size(),f); fclose(f);
    printf("bias=%.3f PCF=3x3 cascades=2\n", bias);
    printf("✓ Shadow Mapping 完成。\n");
    return 0;
}
