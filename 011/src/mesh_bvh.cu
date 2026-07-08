// 011 | Möller-Trumbore + 简易 BVH；程序化“兔子近似”网格
#include "rt_math.cuh"
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstdio>

struct Tri { Vec3 v0,v1,v2; Vec3 n; Vec3 albedo; };
struct AABB { Vec3 bmin, bmax; };
struct BVHNode {
    AABB box;
    int left, right; // 子节点；叶：left=-1, right=tri_index 或 tri 范围
    int tri_start, tri_count; // 叶节点三角形范围
};

__host__ __device__ bool intersect_aabb(Ray r, AABB b, float tmin, float tmax) {
    for (int a = 0; a < 3; ++a) {
        float ro = (&r.o.x)[a], rd = (&r.d.x)[a];
        float inv = 1.f / (fabsf(rd) < 1e-8f ? copysignf(1e-8f, rd) : rd);
        float t0 = ((&b.bmin.x)[a] - ro) * inv;
        float t1 = ((&b.bmax.x)[a] - ro) * inv;
        if (inv < 0) { float tmp=t0; t0=t1; t1=tmp; }
        tmin = t0 > tmin ? t0 : tmin;
        tmax = t1 < tmax ? t1 : tmax;
        if (tmax <= tmin) return false;
    }
    return true;
}

// Möller–Trumbore 光线-三角形求交（核心：边叉乘求重心坐标 u,v 与 t）
// 若 det 接近 0 则光线平行于三角形平面，加速结构依赖此精确求交
__host__ __device__ bool intersect_tri(const Tri& tri, Ray r, float tmin, float tmax, float& t, Vec3& n) {
    const float EPS = 1e-6f;
    Vec3 e1 = tri.v1 - tri.v0;
    Vec3 e2 = tri.v2 - tri.v0;
    Vec3 pvec = cross(r.d, e2);
    float det = dot(e1, pvec);
    if (fabsf(det) < EPS) return false;
    float invDet = 1.f / det;
    Vec3 tvec = r.o - tri.v0;
    float u = dot(tvec, pvec) * invDet;
    if (u < 0 || u > 1) return false;
    Vec3 qvec = cross(tvec, e1);
    float v = dot(r.d, qvec) * invDet;
    if (v < 0 || u + v > 1) return false;
    float tt = dot(e2, qvec) * invDet;
    if (tt < tmin || tt > tmax) return false;
    t = tt; n = tri.n; return true;
}

AABB tri_bounds(const Tri& t) {
    AABB b;
    b.bmin = Vec3{fminf(t.v0.x,fminf(t.v1.x,t.v2.x)), fminf(t.v0.y,fminf(t.v1.y,t.v2.y)), fminf(t.v0.z,fminf(t.v1.z,t.v2.z))};
    b.bmax = Vec3{fmaxf(t.v0.x,fmaxf(t.v1.x,t.v2.x)), fmaxf(t.v0.y,fmaxf(t.v1.y,t.v2.y)), fmaxf(t.v0.z,fmaxf(t.v1.z,t.v2.z))};
    return b;
}
AABB merge_box(AABB a, AABB b) {
    return {Vec3{fminf(a.bmin.x,b.bmin.x),fminf(a.bmin.y,b.bmin.y),fminf(a.bmin.z,b.bmin.z)},
            Vec3{fmaxf(a.bmax.x,b.bmax.x),fmaxf(a.bmax.y,b.bmax.y),fmaxf(a.bmax.z,b.bmax.z)}};
}

// 递归构建 BVH（中位数劈分最长轴；叶节点存三角形区间）
// 目的：用层次包围盒加速光线遍历，减少无效三角测试
int build_bvh(std::vector<BVHNode>& nodes, std::vector<Tri>& tris, std::vector<int>& idx, int start, int end) {
    BVHNode node{};
    node.left = node.right = -1;
    node.tri_start = start; node.tri_count = end - start;
    AABB box = tri_bounds(tris[idx[start]]);
    for (int i = start+1; i < end; ++i) box = merge_box(box, tri_bounds(tris[idx[i]]));
    node.box = box;
    int id = (int)nodes.size();
    nodes.push_back(node);
    if (end - start <= 4) return id; // 叶
    // 最长轴
    Vec3 ext = box.bmax - box.bmin;
    int axis = 0;
    if (ext.y > ext.x) axis = 1;
    if (ext.z > (axis==0?ext.x:ext.y)) axis = 2;
    int mid = (start + end) / 2;
    std::nth_element(idx.begin()+start, idx.begin()+mid, idx.begin()+end, [&](int a, int b){
        Vec3 ca = (tris[a].v0+tris[a].v1+tris[a].v2)*(1.f/3.f);
        Vec3 cb = (tris[b].v0+tris[b].v1+tris[b].v2)*(1.f/3.f);
        return (&ca.x)[axis] < (&cb.x)[axis];
    });
    int L = build_bvh(nodes, tris, idx, start, mid);
    int R = build_bvh(nodes, tris, idx, mid, end);
    nodes[id].left = L; nodes[id].right = R;
    nodes[id].tri_count = 0; // 内部节点
    return id;
}

// 生成简化 Stanford Bunny 近似：椭球 + 耳
void make_bunny_like(std::vector<Tri>& tris, int slices=24, int stacks=16) {
    auto add_sphere = [&](Vec3 c, float rx, float ry, float rz, Vec3 alb) {
        for (int i = 0; i < stacks; ++i) {
            float v0 = (float)i / stacks, v1 = (float)(i+1)/stacks;
            float phi0 = v0 * 3.14159265f, phi1 = v1 * 3.14159265f;
            for (int j = 0; j < slices; ++j) {
                float u0 = (float)j / slices, u1 = (float)(j+1)/slices;
                float th0 = u0 * 6.2831853f, th1 = u1 * 6.2831853f;
                auto sph = [&](float th, float phi) {
                    return Vec3{c.x + rx*sinf(phi)*cosf(th), c.y + ry*cosf(phi), c.z + rz*sinf(phi)*sinf(th)};
                };
                Vec3 p00=sph(th0,phi0), p10=sph(th1,phi0), p01=sph(th0,phi1), p11=sph(th1,phi1);
                auto push = [&](Vec3 a, Vec3 b, Vec3 c3) {
                    Vec3 n = normalize(cross(b-a, c3-a));
                    tris.push_back({a,b,c3,n,alb});
                };
                if (i) push(p00,p10,p11);
                if (i < stacks-1) push(p00,p11,p01);
            }
        }
    };
    add_sphere(Vec3{0,0.2f,-1.5f}, 0.45f, 0.4f, 0.4f, Vec3{0.85f,0.75f,0.65f}); // 身体
    add_sphere(Vec3{0,0.55f,-1.35f}, 0.22f, 0.2f, 0.22f, Vec3{0.9f,0.8f,0.7f}); // 头
    add_sphere(Vec3{-0.12f,0.85f,-1.3f}, 0.06f, 0.18f, 0.06f, Vec3{0.9f,0.8f,0.7f}); // 耳
    add_sphere(Vec3{0.12f,0.85f,-1.3f}, 0.06f, 0.18f, 0.06f, Vec3{0.9f,0.8f,0.7f});
}

__global__ void render_naive(const Tri* tris, int ntri, uint8_t* out, int W, int H) {
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;
    if (x>=W||y>=H) return;
    float aspect=(float)W/H;
    float u=(2.f*(x+.5f)/W-1.f)*aspect, v=1.f-2.f*(y+.5f)/H;
    Ray r{Vec3{0,0.5f,1.2f}, normalize(Vec3{u,v-0.2f,-1.5f})};
    float best=1e20f; Vec3 n, alb;
    for (int i=0;i<ntri;++i) {
        float t; Vec3 nn;
        if (intersect_tri(tris[i], r, 0.001f, best, t, nn)) { best=t; n=nn; alb=tris[i].albedo; }
    }
    Vec3 col{0.15f,0.18f,0.25f};
    if (best < 1e19f) {
        float ndl = fmaxf(0.f, dot(n, normalize(Vec3{1,1,0.5f})));
        col = alb * (0.2f + 0.8f*ndl);
    }
    int i=(y*W+x)*3;
    out[i]=(uint8_t)(fminf(1.f,col.x)*255);
    out[i+1]=(uint8_t)(fminf(1.f,col.y)*255);
    out[i+2]=(uint8_t)(fminf(1.f,col.z)*255);
}

__global__ void render_bvh(const Tri* tris, const int* order, const BVHNode* nodes, int root,
                           uint8_t* out, int W, int H) {
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;
    if (x>=W||y>=H) return;
    float aspect=(float)W/H;
    float u=(2.f*(x+.5f)/W-1.f)*aspect, v=1.f-2.f*(y+.5f)/H;
    Ray r{Vec3{0,0.5f,1.2f}, normalize(Vec3{u,v-0.2f,-1.5f})};
    // 栈式 BVH 遍历（设备端无递归）：先测 AABB 包围盒，命中再下钻或测试三角形
    int stack[64]; int sp=0; stack[sp++]=root;
    float best=1e20f; Vec3 n, alb;
    while (sp > 0) {
        int ni = stack[--sp];
        BVHNode node = nodes[ni];
        if (!intersect_aabb(r, node.box, 0.001f, best)) continue;
        if (node.tri_count > 0) {
            for (int i=0;i<node.tri_count;++i) {
                const Tri& t = tris[order[node.tri_start+i]];
                float tt; Vec3 nn;
                if (intersect_tri(t, r, 0.001f, best, tt, nn)) { best=tt; n=nn; alb=t.albedo; }
            }
        } else {
            if (node.left >= 0) stack[sp++] = node.left;
            if (node.right >= 0) stack[sp++] = node.right;
        }
    }
    Vec3 col{0.15f,0.18f,0.25f};
    if (best < 1e19f) {
        float ndl = fmaxf(0.f, dot(n, normalize(Vec3{1,1,0.5f})));
        col = alb * (0.2f + 0.8f*ndl);
    }
    int i=(y*W+x)*3;
    out[i]=(uint8_t)(fminf(1.f,col.x)*255);
    out[i+1]=(uint8_t)(fminf(1.f,col.y)*255);
    out[i+2]=(uint8_t)(fminf(1.f,col.z)*255);
}

int main() {
    printf("============================================================\n");
    printf("011 | 三角形网格 + BVH (Möller-Trumbore)\n");
    printf("============================================================\n");
    std::vector<Tri> tris;
    make_bunny_like(tris, 32, 20);
    printf("三角形数: %zu\n", tris.size());
    std::vector<int> idx(tris.size());
    for (size_t i=0;i<idx.size();++i) idx[i]=(int)i;
    std::vector<BVHNode> nodes;
    int root = build_bvh(nodes, tris, idx, 0, (int)idx.size());
    printf("BVH 节点数: %zu root=%d\n", nodes.size(), root);

    const int W=640, H=480;
    Tri* d_tris; int* d_ord; BVHNode* d_nodes; uint8_t* d_out;
    cudaMalloc(&d_tris, tris.size()*sizeof(Tri));
    cudaMalloc(&d_ord, idx.size()*sizeof(int));
    cudaMalloc(&d_nodes, nodes.size()*sizeof(BVHNode));
    cudaMalloc(&d_out, W*H*3);
    cudaMemcpy(d_tris, tris.data(), tris.size()*sizeof(Tri), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ord, idx.data(), idx.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes, nodes.data(), nodes.size()*sizeof(BVHNode), cudaMemcpyHostToDevice);
    dim3 block(16,16), grid((W+15)/16,(H+15)/16);

    auto bench = [&](auto launch, int reps) {
        launch(); cudaDeviceSynchronize();
        auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<reps;++i) launch();
        cudaDeviceSynchronize();
        auto t1=std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double,std::milli>(t1-t0).count()/reps;
    };
    double ms_naive = bench([&]{ render_naive<<<grid,block>>>(d_tris,(int)tris.size(),d_out,W,H); }, 5);
    double ms_bvh = bench([&]{ render_bvh<<<grid,block>>>(d_tris,d_ord,d_nodes,root,d_out,W,H); }, 20);
    printf("无 BVH: %.3f ms | 有 BVH: %.3f ms | 加速: %.2fx\n", ms_naive, ms_bvh, ms_naive/ms_bvh);
    std::vector<uint8_t> img(W*H*3);
    cudaMemcpy(img.data(), d_out, img.size(), cudaMemcpyDeviceToHost);
    write_ppm("output/bunny_bvh.ppm", img, W, H);
    printf("✓ Möller-Trumbore + BVH 完成。\n");
    return 0;
}
