// 018 | 软件光栅演示完整管线：MVP + Phong 片元着色，离线写 PPM
// 对应 GLSL 在 shaders/；参数文件模拟 ImGui 调参
#include <cmath>
#include <cstdio>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>

struct V3 { float x,y,z; };
struct V4 { float x,y,z,w; };
struct M4 { float m[16]; }; // column-major

static V3 operator+(V3 a,V3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
static V3 operator-(V3 a,V3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
static V3 operator*(V3 a,float s){return {a.x*s,a.y*s,a.z*s};}
static float dot(V3 a,V3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static V3 cross(V3 a,V3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
static V3 normalize(V3 v){float L=sqrtf(dot(v,v));return L>0?v*(1.f/L):v;}

static M4 mul4(M4 a,M4 b){
    M4 r{};
    for(int c=0;c<4;++c) for(int row=0;row<4;++row){
        r.m[c*4+row]=0;
        for(int k=0;k<4;++k) r.m[c*4+row]+=a.m[k*4+row]*b.m[c*4+k];
    }
    return r;
}
static V4 mul(M4 m, V4 v){
    return {
        m.m[0]*v.x+m.m[4]*v.y+m.m[8]*v.z+m.m[12]*v.w,
        m.m[1]*v.x+m.m[5]*v.y+m.m[9]*v.z+m.m[13]*v.w,
        m.m[2]*v.x+m.m[6]*v.y+m.m[10]*v.z+m.m[14]*v.w,
        m.m[3]*v.x+m.m[7]*v.y+m.m[11]*v.z+m.m[15]*v.w
    };
}
static M4 perspective(float fovy, float aspect, float n, float f){
    float t=tanf(fovy*0.5f*3.14159265f/180.f);
    M4 m{}; m.m[0]=1/(aspect*t); m.m[5]=1/t; m.m[10]=-(f+n)/(f-n); m.m[11]=-1; m.m[14]=-2*f*n/(f-n);
    return m;
}
static M4 lookAt(V3 e, V3 c, V3 up){
    V3 z=normalize(e-c), x=normalize(cross(up,z)), y=cross(z,x);
    M4 m{};
    m.m[0]=x.x;m.m[4]=x.y;m.m[8]=x.z;
    m.m[1]=y.x;m.m[5]=y.y;m.m[9]=y.z;
    m.m[2]=z.x;m.m[6]=z.y;m.m[10]=z.z;
    m.m[12]=-dot(x,e); m.m[13]=-dot(y,e); m.m[14]=-dot(z,e); m.m[15]=1;
    return m;
}
static M4 rotateY(float a){
    float c=cosf(a),s=sinf(a); M4 m{};
    m.m[0]=c;m.m[8]=s;m.m[5]=1;m.m[2]=-s;m.m[10]=c;m.m[15]=1; return m;
}

// 立方体 12 三角形
static void cube_mesh(std::vector<V3>& pos, std::vector<V3>& nrm, std::vector<int>& idx){
    V3 f[6][4]={
        {{1,-1,-1},{1,1,-1},{1,1,1},{1,-1,1}},
        {{-1,-1,1},{-1,1,1},{-1,1,-1},{-1,-1,-1}},
        {{-1,1,-1},{-1,1,1},{1,1,1},{1,1,-1}},
        {{-1,-1,1},{-1,-1,-1},{1,-1,-1},{1,-1,1}},
        {{-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1}},
        {{1,-1,-1},{-1,-1,-1},{-1,1,-1},{1,1,-1}},
    };
    V3 ns[6]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    for(int i=0;i<6;++i){
        int b=(int)pos.size();
        for(int k=0;k<4;++k){ pos.push_back(f[i][k]*0.5f); nrm.push_back(ns[i]); }
        idx.push_back(b);idx.push_back(b+1);idx.push_back(b+2);
        idx.push_back(b);idx.push_back(b+2);idx.push_back(b+3);
    }
}

// 片元 Phong（对应 phong.frag）
static V3 phong(V3 world, V3 n, V3 light, V3 view, V3 albedo){
    V3 N=normalize(n), L=normalize(light-world), V=normalize(view-world);
    V3 R = L*(-1.f) + N*(2.f*dot(N,L)); // reflect(-L,N) with L toward light
    R = normalize(L*(-1)+N*(2*fmaxf(0.f,dot(N,L)))); // safer
    // 标准 reflect(-L,N) where L is light direction from point
    L = normalize(light-world);
    R = normalize( (L*(-1)) + N * (2.f*dot(N,L)) );
    float diff=fmaxf(0.f,dot(N,L));
    float spec=powf(fmaxf(0.f,dot(R,V)), 32.f);
    return albedo*(0.15f+0.7f*diff) + V3{1,1,1}*(0.4f*spec);
}

struct Params { float angle; float light_x; V3 albedo; };

static Params load_params(const char* path){
    // 模拟 ImGui 热调：从文件读
    Params p{0.7f, 2.f, {0.8f,0.3f,0.2f}};
    std::ifstream in(path);
    if(!in) return p;
    std::string k; float v;
    while(in>>k>>v){
        if(k=="angle") p.angle=v;
        if(k=="light_x") p.light_x=v;
        if(k=="albedo_r") p.albedo.x=v;
        if(k=="albedo_g") p.albedo.y=v;
        if(k=="albedo_b") p.albedo.z=v;
    }
    return p;
}

int main(){
    printf("============================================================\n");
    printf("018 | 软件管线：旋转立方体 + Phong (离线)\n");
    printf("============================================================\n");
    const int W=640,H=480;
    Params P = load_params("params.txt");
    std::vector<V3> pos,nrm; std::vector<int> idx;
    cube_mesh(pos,nrm,idx);
    M4 model=rotateY(P.angle);
    M4 view=lookAt({0,0.8f,2.5f},{0,0,0},{0,1,0});
    M4 proj=perspective(45.f,(float)W/H,0.1f,100.f);
    M4 mvp=mul4(proj,mul4(view,model));
    std::vector<float> zbuf(W*H,1e30f);
    std::vector<unsigned char> img(W*H*3, 20);
    V3 light{P.light_x,2.f,2.f}, eye{0,0.8f,2.5f};

    // 顶点着色 + 光栅 + 片元
    for(size_t t=0;t<idx.size();t+=3){
        V3 wp[3]; V3 wn[3]; V4 clip[3]; float sx[3],sy[3];
        for(int k=0;k<3;++k){
            int id=idx[t+k];
            V4 wo=mul(model,{pos[id].x,pos[id].y,pos[id].z,1});
            wp[k]={wo.x,wo.y,wo.z};
            V4 no=mul(model,{nrm[id].x,nrm[id].y,nrm[id].z,0});
            wn[k]=normalize({no.x,no.y,no.z});
            clip[k]=mul(mvp,{pos[id].x,pos[id].y,pos[id].z,1});
            float iw=1.f/clip[k].w;
            sx[k]=(clip[k].x*iw*0.5f+0.5f)*W;
            sy[k]=(1.f-(clip[k].y*iw*0.5f+0.5f))*H;
        }
        // 包围盒
        int minx=std::max(0,(int)std::floor(std::min({sx[0],sx[1],sx[2]})));
        int maxx=std::min(W-1,(int)std::ceil(std::max({sx[0],sx[1],sx[2]})));
        int miny=std::max(0,(int)std::floor(std::min({sy[0],sy[1],sy[2]})));
        int maxy=std::min(H-1,(int)std::ceil(std::max({sy[0],sy[1],sy[2]})));
        float area=(sx[1]-sx[0])*(sy[2]-sy[0])-(sx[2]-sx[0])*(sy[1]-sy[0]);
        if(fabsf(area)<1e-6f) continue;
        for(int y=miny;y<=maxy;++y) for(int x=minx;x<=maxx;++x){
            float px=x+0.5f, py=y+0.5f;
            float w0=((sx[1]-px)*(sy[2]-py)-(sx[2]-px)*(sy[1]-py))/area;
            float w1=((sx[2]-px)*(sy[0]-py)-(sx[0]-px)*(sy[2]-py))/area;
            float w2=1-w0-w1;
            if(w0<0||w1<0||w2<0) continue;
            float z=w0*clip[0].z/clip[0].w + w1*clip[1].z/clip[1].w + w2*clip[2].z/clip[2].w;
            int p=y*W+x; if(z>=zbuf[p]) continue; zbuf[p]=z;
            V3 world=wp[0]*w0+wp[1]*w1+wp[2]*w2;
            V3 normal=normalize(wn[0]*w0+wn[1]*w1+wn[2]*w2);
            V3 c=phong(world,normal,light,eye,P.albedo);
            img[p*3]=(unsigned char)(std::min(1.f,std::max(0.f,c.x))*255);
            img[p*3+1]=(unsigned char)(std::min(1.f,std::max(0.f,c.y))*255);
            img[p*3+2]=(unsigned char)(std::min(1.f,std::max(0.f,c.z))*255);
        }
    }
    FILE* f=fopen("output/cube.ppm","wb");
    fprintf(f,"P6\n%d %d\n255\n",W,H); fwrite(img.data(),1,img.size(),f); fclose(f);
    // 写出默认 params 供“ImGui”式调参
    std::ofstream pf("params.txt");
    pf<<"angle "<<P.angle<<"\nlight_x "<<P.light_x<<"\nalbedo_r "<<P.albedo.x<<"\nalbedo_g "<<P.albedo.y<<"\nalbedo_b "<<P.albedo.z<<"\n";
    printf("管线: 顶点 → MVP(VS) → 光栅 → Phong(FS) → PPM\n");
    printf("已写 output/cube.ppm，调参文件 params.txt\n");
    printf("✓ 完整渲染管线离线演示完成。\n");
    return 0;
}
