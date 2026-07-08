// 025 | 迷你引擎：场景图、剔除、多 pass、相机、统计
#include <cmath>
#include <cstdio>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <memory>
struct V3{float x,y,z;};
static V3 operator+(V3 a,V3 b){return{a.x+b.x,a.y+b.y,a.z+b.z};}
static V3 operator-(V3 a,V3 b){return{a.x-b.x,a.y-b.y,a.z-b.z};}
static V3 operator*(V3 a,float s){return{a.x*s,a.y*s,a.z*s};}
static float dot(V3 a,V3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static float len(V3 v){return sqrtf(dot(v,v));}
static V3 normalize(V3 v){float L=len(v);return L>0?v*(1/L):v;}

struct AABB{V3 bmin,bmax;};
struct Node {
    std::string name; V3 pos; float radius; V3 albedo; float metal, rough;
    std::vector<std::unique_ptr<Node>> children;
    AABB world_bounds() const {
        return {{pos.x-radius,pos.y-radius,pos.z-radius},{pos.x+radius,pos.y+radius,pos.z+radius}};
    }
};

struct Camera { V3 pos, front, up; float yaw, pitch;
    void update(){
        front=normalize({cosf(yaw)*cosf(pitch), sinf(pitch), sinf(yaw)*cosf(pitch)});
    }
};

struct Stats { int draw_calls=0; int culled=0; double ms_shadow=0, ms_gbuf=0, ms_light=0, ms_post=0; };

static bool frustum_aabb(const AABB& b, V3 eye, V3 dir, float farp){
    // 简化：包围球中心是否在前方
    V3 c{(b.bmin.x+b.bmax.x)*0.5f,(b.bmin.y+b.bmax.y)*0.5f,(b.bmin.z+b.bmax.z)*0.5f};
    return dot(c-eye, dir) > -2.f && len(c-eye) < farp;
}

static void collect_visible(Node* n, Camera& cam, std::vector<Node*>& out, Stats& st){
    if(!n) return;
    if(!frustum_aabb(n->world_bounds(), cam.pos, cam.front, 50.f)){ st.culled++; return; }
    out.push_back(n); st.draw_calls++;
    for(auto& c: n->children) collect_visible(c.get(), cam, out, st);
}

int main(){
    printf("============================================================\n");
    printf("025 | 迷你引擎渲染器（无头帧输出）\n");
    printf("============================================================\n");
    // 场景图：根 → 若干物体（Sponza 用球阵列代替）
    auto root=std::make_unique<Node>(); root->name="root"; root->radius=0;
    for(int i=0;i<20;++i){
        auto n=std::make_unique<Node>();
        n->name="mesh_"+std::to_string(i);
        n->pos={sinf(i*0.7f)*3.f, 0.4f, -2.f-i*0.3f};
        n->radius=0.3f+0.1f*(i%3);
        n->albedo={0.5f+0.4f*sinf(i),0.4f,0.3f+0.4f*cosf(i)};
        n->metal=(i%4==0)?0.8f:0.f; n->rough=0.2f+0.1f*(i%5);
        root->children.push_back(std::move(n));
    }
    Camera cam{{0,1.2f,3.f},{0,0,-1},{0,1,0}, -1.57f, -0.2f}; cam.update();
    Stats st;
    std::vector<Node*> visible;
    auto t0=std::chrono::high_resolution_clock::now();
    collect_visible(root.get(), cam, visible, st);
    auto t1=std::chrono::high_resolution_clock::now();
    st.ms_gbuf=std::chrono::duration<double,std::milli>(t1-t0).count();

    const int W=640,H=360;
    std::vector<unsigned char> img(W*H*3, 25);
    // 光照 pass（对可见物体）
    t0=std::chrono::high_resolution_clock::now();
    for(int y=0;y<H;++y)for(int x=0;x<W;++x){
        float u=(x+.5f)/W*2-1, v=1-(y+.5f)/H*2;
        V3 right=normalize({cam.front.z,0,-cam.front.x});
        V3 rd=normalize(cam.front + right*u*((float)W/H*0.6f) + cam.up*v*0.6f);
        float best=1e30f; V3 col{0.12f,0.14f,0.18f}; V3 n; V3 alb; float metal=0,rough=1; bool hit=false;
        for(Node* o: visible){
            if(o->radius<=0) continue;
            V3 oc=cam.pos-o->pos; float a=dot(rd,rd),b=2*dot(oc,rd),c=dot(oc,oc)-o->radius*o->radius;
            float d=b*b-4*a*c; if(d<0)continue; float t=(-b-sqrtf(d))/(2*a);
            if(t>0.01f&&t<best){best=t;n=normalize(cam.pos+rd*t-o->pos);alb=o->albedo;metal=o->metal;rough=o->rough;hit=true;}
        }
        if(fabsf(rd.y)>1e-6f){float t=-cam.pos.y/rd.y; if(t>0.01f&&t<best){best=t;n={0,1,0};alb={0.5f,0.5f,0.5f};metal=0;rough=0.9f;hit=true;}}
        if(hit){
            V3 L=normalize(V3{1,2,1}); float ndl=fmaxf(0.f,dot(n,L));
            // 简易 PBR 混合
            col = alb*((1-metal)*(0.15f+0.7f*ndl) + metal*powf(ndl, 1.f/fmaxf(0.04f,rough))*0.8f);
        }
        int i=(y*W+x)*3;
        img[i]=(unsigned char)(std::min(1.f,col.x)*255);
        img[i+1]=(unsigned char)(std::min(1.f,col.y)*255);
        img[i+2]=(unsigned char)(std::min(1.f,col.z)*255);
    }
    t1=std::chrono::high_resolution_clock::now();
    st.ms_light=std::chrono::duration<double,std::milli>(t1-t0).count();
    // 后处理：简单 gamma
    t0=std::chrono::high_resolution_clock::now();
    for(size_t i=0;i<img.size();++i){ float v=img[i]/255.f; img[i]=(unsigned char)(sqrtf(v)*255); }
    t1=std::chrono::high_resolution_clock::now();
    st.ms_post=std::chrono::duration<double,std::milli>(t1-t0).count();

    FILE* f=fopen("output/frame.ppm","wb"); fprintf(f,"P6\n%d %d\n255\n",W,H); fwrite(img.data(),1,img.size(),f); fclose(f);
    std::ofstream stats("output/stats.txt");
    stats<<"draw_calls "<<st.draw_calls<<"\nculled "<<st.culled<<"\n";
    stats<<"ms_cull_gbuf "<<st.ms_gbuf<<"\nms_light "<<st.ms_light<<"\nms_post "<<st.ms_post<<"\n";
    stats<<"camera_pos "<<cam.pos.x<<" "<<cam.pos.y<<" "<<cam.pos.z<<"\n";
    printf("draw_calls=%d culled=%d light=%.2fms post=%.2fms\n", st.draw_calls, st.culled, st.ms_light, st.ms_post);
    printf("✓ 迷你引擎无头帧 + 统计完成（见 docs/vulkan_pipeline.md）。\n");
    return 0;
}
