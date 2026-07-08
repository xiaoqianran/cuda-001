// 022 | Deferred Shading：G-Buffer pass + 光照 pass + 透明前向混合
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>
struct V3{float x,y,z;};
static V3 operator+(V3 a,V3 b){return{a.x+b.x,a.y+b.y,a.z+b.z};}
static V3 operator-(V3 a,V3 b){return{a.x-b.x,a.y-b.y,a.z-b.z};}
static V3 operator*(V3 a,float s){return{a.x*s,a.y*s,a.z*s};}
static V3 operator*(V3 a,V3 b){return{a.x*b.x,a.y*b.y,a.z*b.z};}
static float dot(V3 a,V3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static V3 normalize(V3 v){float L=sqrtf(dot(v,v));return L>0?v*(1/L):v;}

struct Sphere{V3 c;float r;V3 alb;float metal,rough;bool transparent;};
struct Light{V3 p;V3 col;float radius;};
struct GBuffer {
    std::vector<float> pos,nrm,albedo,mr,depth; int w,h;
};

static void gbuffer_pass(GBuffer& g, const std::vector<Sphere>& scene, V3 eye){
    int W=g.w,H=g.h;
    g.pos.assign(W*H*3,0); g.nrm.assign(W*H*3,0); g.albedo.assign(W*H*3,0);
    g.mr.assign(W*H*2,0); g.depth.assign(W*H,1e30f);
    for(int y=0;y<H;++y)for(int x=0;x<W;++x){
        float u=(x+.5f)/W*2-1, v=1-(y+.5f)/H*2;
        V3 rd=normalize({u*(float)W/H,v,-1.5f});
        float best=1e30f; int hit=-1; V3 n{};
        for(int i=0;i<(int)scene.size();++i){
            if(scene[i].transparent) continue;
            V3 oc=eye-scene[i].c;
            float aa=dot(rd,rd), bb=2*dot(oc,rd), cc=dot(oc,oc)-scene[i].r*scene[i].r;
            float disc=bb*bb-4*aa*cc; if(disc<0) continue;
            float t=(-bb-sqrtf(disc))/(2*aa);
            if(t>0.01f&&t<best){best=t;hit=i;n=normalize(eye+rd*t-scene[i].c);}
        }
        if(fabsf(rd.y)>1e-6f){float t=-eye.y/rd.y; if(t>0.01f&&t<best){best=t;hit=-2;n={0,1,0};}}
        int i=y*W+x;
        if(hit==-1) continue;
        V3 p=eye+rd*best; g.depth[i]=best;
        g.pos[i*3]=p.x;g.pos[i*3+1]=p.y;g.pos[i*3+2]=p.z;
        g.nrm[i*3]=n.x;g.nrm[i*3+1]=n.y;g.nrm[i*3+2]=n.z;
        if(hit==-2){ g.albedo[i*3]=g.albedo[i*3+1]=g.albedo[i*3+2]=0.55f; g.mr[i*2]=0;g.mr[i*2+1]=0.9f; }
        else { auto& s=scene[hit]; g.albedo[i*3]=s.alb.x;g.albedo[i*3+1]=s.alb.y;g.albedo[i*3+2]=s.alb.z; g.mr[i*2]=s.metal;g.mr[i*2+1]=s.rough; }
    }
}

static void light_pass(const GBuffer& g, const std::vector<Light>& lights, std::vector<unsigned char>& img){
    int W=g.w,H=g.h; img.assign(W*H*3,0);
    for(int y=0;y<H;++y)for(int x=0;x<W;++x){
        int i=y*W+x;
        if(g.depth[i]>1e29f){ img[i*3]=20;img[i*3+1]=22;img[i*3+2]=30; continue; }
        V3 p{g.pos[i*3],g.pos[i*3+1],g.pos[i*3+2]};
        V3 n=normalize({g.nrm[i*3],g.nrm[i*3+1],g.nrm[i*3+2]});
        V3 alb{g.albedo[i*3],g.albedo[i*3+1],g.albedo[i*3+2]};
        V3 col=alb*0.05f;
        for(const auto& L: lights){
            V3 toL=L.p-p; float dist=sqrtf(dot(toL,toL));
            if(dist>L.radius) continue;
            V3 l=toL*(1.f/dist);
            float atten=1.f/(1.f+0.1f*dist+0.02f*dist*dist);
            float ndl=fmaxf(0.f,dot(n,l));
            float fall=1.f-dist/L.radius; fall*=fall;
            col = col + alb * L.col * (ndl*atten*fall);
        }
        img[i*3]=(unsigned char)(std::min(1.f,col.x)*255);
        img[i*3+1]=(unsigned char)(std::min(1.f,col.y)*255);
        img[i*3+2]=(unsigned char)(std::min(1.f,col.z)*255);
    }
}

static void forward_transparent(std::vector<unsigned char>& img, int W, int H, V3 eye, const Sphere& s){
    for(int y=0;y<H;++y)for(int x=0;x<W;++x){
        float u=(x+.5f)/W*2-1, v=1-(y+.5f)/H*2;
        V3 rd=normalize({u*(float)W/H,v,-1.5f});
        V3 oc=eye-s.c;
        float aa=dot(rd,rd), bb=2*dot(oc,rd), cc=dot(oc,oc)-s.r*s.r;
        float disc=bb*bb-4*aa*cc; if(disc<0) continue;
        float t=(-bb-sqrtf(disc))/(2*aa); if(t<0.01f) continue;
        V3 n=normalize(eye+rd*t-s.c);
        float fres=powf(1.f-fmaxf(0.f,dot(n,rd*(-1.f))),3.f);
        float alpha=0.35f+0.4f*fres;
        int i=(y*W+x)*3;
        float rr=img[i]/255.f, gg=img[i+1]/255.f, bbv=img[i+2]/255.f;
        rr=rr*(1-alpha)+s.alb.x*alpha; gg=gg*(1-alpha)+s.alb.y*alpha; bbv=bbv*(1-alpha)+s.alb.z*alpha;
        img[i]=(unsigned char)(rr*255); img[i+1]=(unsigned char)(gg*255); img[i+2]=(unsigned char)(bbv*255);
    }
}

int main(){
    printf("============================================================\n");
    printf("022 | Deferred Shading (G-Buffer + 多光源)\n");
    printf("============================================================\n");
    const int W=640,H=360;
    std::vector<Sphere> scene={
        {{0,0.5f,-2},0.5f,{0.9f,0.2f,0.2f},0.1f,0.4f,false},
        {{-1.2f,0.4f,-2.5f},0.4f,{0.2f,0.8f,0.3f},0.0f,0.7f,false},
        {{1.1f,0.4f,-2.2f},0.4f,{0.2f,0.3f,0.9f},0.8f,0.2f,false},
        {{0.3f,0.35f,-1.5f},0.3f,{0.8f,0.9f,1.0f},0,0,true},
    };
    std::vector<Light> lights;
    for(int i=0;i<32;++i){
        float a=i*0.4f;
        lights.push_back({{sinf(a)*2.f,1.2f+0.3f*cosf(a*2),-2.f+cosf(a)*1.5f},
            {0.5f+0.5f*sinf(a),0.5f+0.5f*cosf(a),0.8f},3.5f});
    }
    GBuffer g; g.w=W; g.h=H;
    V3 eye{0,1.0f,1.5f};
    gbuffer_pass(g, scene, eye);
    std::vector<unsigned char> img;
    light_pass(g, lights, img);
    for(auto& s: scene) if(s.transparent) forward_transparent(img,W,H,eye,s);
    FILE* f=fopen("output/deferred.ppm","wb");
    fprintf(f,"P6\n%d %d\n255\n",W,H); fwrite(img.data(),1,img.size(),f); fclose(f);
    printf("光源数: %zu | G-Buffer: pos/normal/albedo/metal-rough\n", lights.size());
    printf("✓ 延迟渲染管线完成。\n");
    return 0;
}
