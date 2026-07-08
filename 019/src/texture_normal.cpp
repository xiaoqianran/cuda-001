// 019 | 纹理 + 法线贴图：程序化生成贴图，TBN 着色，Mipmap 过滤对比
#include <cmath>
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>

struct V3{float x,y,z;};
static float dot(V3 a,V3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static V3 operator+(V3 a,V3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
static V3 operator-(V3 a,V3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
static V3 operator*(V3 a,float s){return {a.x*s,a.y*s,a.z*s};}
static V3 normalize(V3 v){float L=sqrtf(dot(v,v));return L>0?v*(1/L):v;}
static V3 cross(V3 a,V3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}

struct Tex { int w,h; std::vector<float> rgba; }; // 浮点 0-1

static Tex make_albedo(int n){
    Tex t{n,n,std::vector<float>(n*n*4)};
    for(int y=0;y<n;++y)for(int x=0;x<n;++x){
        int i=(y*n+x)*4;
        float u=x/(float)n,v=y/(float)n;
        // 棋盘砖块
        int cx=(int)(u*8), cy=(int)(v*8);
        float c=((cx+cy)&1)?0.85f:0.45f;
        t.rgba[i]=c; t.rgba[i+1]=c*0.7f; t.rgba[i+2]=c*0.5f; t.rgba[i+3]=1;
    }
    return t;
}
static Tex make_spec(int n){
    Tex t{n,n,std::vector<float>(n*n*4)};
    for(int y=0;y<n;++y)for(int x=0;x<n;++x){
        int i=(y*n+x)*4;
        float s = ((x/32)+(y/32))%2 ? 0.9f : 0.1f;
        t.rgba[i]=t.rgba[i+1]=t.rgba[i+2]=s; t.rgba[i+3]=1;
    }
    return t;
}
// 法线贴图：从高度场生成
static Tex make_normal_from_height(int n){
    std::vector<float> H(n*n);
    for(int y=0;y<n;++y)for(int x=0;x<n;++x){
        float u=x/(float)n,v=y/(float)n;
        H[y*n+x]=0.15f*sinf(u*30)*cosf(v*30);
    }
    Tex t{n,n,std::vector<float>(n*n*4)};
    for(int y=0;y<n;++y)for(int x=0;x<n;++x){
        float hL=H[y*n+(x?x-1:x)], hR=H[y*n+(x+1<n?x+1:x)];
        float hD=H[(y?y-1:y)*n+x], hU=H[(y+1<n?y+1:y)*n+x];
        V3 nrm=normalize({hL-hR, hD-hU, 1.f});
        int i=(y*n+x)*4;
        t.rgba[i]=nrm.x*0.5f+0.5f; t.rgba[i+1]=nrm.y*0.5f+0.5f; t.rgba[i+2]=nrm.z*0.5f+0.5f; t.rgba[i+3]=1;
    }
    return t;
}

// 生成 Mipmap 链
static std::vector<Tex> build_mips(Tex base){
    std::vector<Tex> mips{base};
    while(mips.back().w>1){
        Tex& p=mips.back();
        int nw=std::max(1,p.w/2), nh=std::max(1,p.h/2);
        Tex m{nw,nh,std::vector<float>(nw*nh*4)};
        for(int y=0;y<nh;++y)for(int x=0;x<nw;++x){
            for(int c=0;c<4;++c){
                float s=0;
                for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx)
                    s+=p.rgba[((std::min(p.h-1,y*2+dy))*p.w+std::min(p.w-1,x*2+dx))*4+c];
                m.rgba[(y*nw+x)*4+c]=s*0.25f;
            }
        }
        mips.push_back(m);
    }
    return mips;
}

static V3 sample_bilinear(const Tex& t, float u, float v){
    u=u-floorf(u); v=v-floorf(v);
    float x=u*(t.w-1), y=v*(t.h-1);
    int x0=(int)x, y0=(int)y; int x1=std::min(t.w-1,x0+1), y1=std::min(t.h-1,y0+1);
    float fx=x-x0, fy=y-y0;
    auto at=[&](int X,int Y){int i=(Y*t.w+X)*4; return V3{t.rgba[i],t.rgba[i+1],t.rgba[i+2]};};
    V3 c00=at(x0,y0),c10=at(x1,y0),c01=at(x0,y1),c11=at(x1,y1);
    V3 a=c00*(1-fx)+c10*fx, b=c01*(1-fx)+c11*fx;
    return a*(1-fy)+b*fy;
}

// 平面网格带切线，用法线贴图着色
int main(){
    printf("============================================================\n");
    printf("019 | 纹理 + 法线贴图 + Mipmap + TBN\n");
    printf("============================================================\n");
    auto albedo=make_albedo(256), spec=make_spec(256), nmap=make_normal_from_height(256);
    auto albedo_mips=build_mips(albedo);
    printf("Mipmap 层数: %zu\n", albedo_mips.size());
    const int W=640,H=480;
    std::vector<unsigned char> img(W*H*3);
    V3 light{1,2,2}, eye{0,1,2};
    // 渲染倾斜平面，展示法线扰动
    for(int y=0;y<H;++y)for(int x=0;x<W;++x){
        float u=x/(float)W, v=y/(float)H;
        // 简单射线与 y=0 平面
        V3 ro=eye, rd=normalize({(u-0.5f)*1.2f, -(v-0.3f), -1.f});
        if(fabsf(rd.y)<1e-5f){ img[(y*W+x)*3]=20; continue; }
        float t=-ro.y/rd.y; if(t<0){ img[(y*W+x)*3]=20; continue; }
        V3 p=ro+rd*t;
        float uu=p.x*0.5f+0.5f, vv=p.z*0.5f+0.5f;
        // 根据距离选 mip
        float dist=sqrtf(dot(p-eye,p-eye));
        int lod=std::min((int)albedo_mips.size()-1, (int)(dist*0.5f));
        V3 alb=sample_bilinear(albedo_mips[lod], uu, vv);
        V3 sp=sample_bilinear(spec, uu, vv);
        V3 nm=sample_bilinear(nmap, uu, vv);
        // 解包法线并 TBN（平面 T=x, B=z, N=y）
        V3 nmap_v={nm.x*2-1, nm.y*2-1, nm.z*2-1};
        V3 T{1,0,0}, B{0,0,1}, N{0,1,0};
        V3 Nw=normalize(T*nmap_v.x + B*nmap_v.y + N*nmap_v.z);
        V3 L=normalize(light-p), V=normalize(eye-p);
        V3 R=normalize( (L*(-1.f)) + Nw*(2.f*dot(Nw,L)) );
        float diff=fmaxf(0.f,dot(Nw,L));
        float specv=powf(fmaxf(0.f,dot(R,V)),32.f)*sp.x;
        V3 c=alb*(0.15f+0.7f*diff)+V3{1,1,1}*specv;
        img[(y*W+x)*3]=(unsigned char)(std::min(1.f,c.x)*255);
        img[(y*W+x)*3+1]=(unsigned char)(std::min(1.f,c.y)*255);
        img[(y*W+x)*3+2]=(unsigned char)(std::min(1.f,c.z)*255);
    }
    // 切换：无贴图 vs 有法线
    FILE* f=fopen("output/textured_normal.ppm","wb");
    fprintf(f,"P6\n%d %d\n255\n",W,H); fwrite(img.data(),1,img.size(),f); fclose(f);
    printf("过滤模式: bilinear + mip 选择；TBN 已应用\n");
    printf("✓ 纹理系统核心演示完成。\n");
    return 0;
}
