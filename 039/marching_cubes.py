#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""039 | Marching Cubes：经典 256 构型简化版（12 边插值 + 三角扇出）+ 实时等值面。"""
from pathlib import Path
import numpy as np
from numba import cuda
from PIL import Image
import math

# 边连接：立方体 12 边的顶点对
EDGE_V = np.array([
    [0,1],[1,2],[2,3],[3,0],
    [4,5],[5,6],[6,7],[7,4],
    [0,4],[1,5],[2,6],[3,7]
], np.int32)
# 顶点偏移
VERT = np.array([
    [0,0,0],[1,0,0],[1,1,0],[0,1,0],
    [0,0,1],[1,0,1],[1,1,1],[0,1,1]
], np.float32)

# 简化：对每个 cube 用 12 边交点构建三角形扇（非完整 256 表，但可提取等值面）
# 完整实现嵌入 edge table 的常用子集

@cuda.jit
def mc_extract(vol, N, iso, verts_out, n_verts, max_v):
    """
    每线程一个体素立方体；边插值产生顶点。
    使用原子计数写入输出缓冲。
    """
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    z = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    if x >= N-1 or y >= N-1 or z >= N-1:
        return
    # 8 角点
    v = cuda.local.array(8, np.float32)
    v[0] = vol[z,y,x]; v[1]=vol[z,y,x+1]; v[2]=vol[z,y+1,x+1]; v[3]=vol[z,y+1,x]
    v[4] = vol[z+1,y,x]; v[5]=vol[z+1,y,x+1]; v[6]=vol[z+1,y+1,x+1]; v[7]=vol[z+1,y+1,x]
    cubeindex = 0
    for i in range(8):
        if v[i] < iso:
            cubeindex |= (1 << i)
    if cubeindex == 0 or cubeindex == 255:
        return
    # 边插值点
    epts = cuda.local.array((12, 3), np.float32)
    edge_hit = cuda.local.array(12, np.int32)
    # 顶点表
    ox = (0,1,1,0,0,1,1,0)
    oy = (0,0,1,1,0,0,1,1)
    oz = (0,0,0,0,1,1,1,1)
    edges_a = (0,1,2,3,4,5,6,7,0,1,2,3)
    edges_b = (1,2,3,0,5,6,7,4,4,5,6,7)
    for e in range(12):
        a = edges_a[e]; b = edges_b[e]
        edge_hit[e] = 0
        if ((cubeindex >> a) & 1) != ((cubeindex >> b) & 1):
            va, vb = v[a], v[b]
            t = (iso - va) / (vb - va + 1e-8)
            epts[e, 0] = x + ox[a] + t * (ox[b]-ox[a])
            epts[e, 1] = y + oy[a] + t * (oy[b]-oy[a])
            epts[e, 2] = z + oz[a] + t * (oz[b]-oz[a])
            edge_hit[e] = 1
    # 收集命中边，每 3 个形成三角（教学简化）
    hit_list = cuda.local.array(12, np.int32)
    nh = 0
    for e in range(12):
        if edge_hit[e]:
            hit_list[nh] = e
            nh += 1
    for t in range(0, nh - 2, 1):
        # 扇出
        e0, e1, e2 = hit_list[0], hit_list[t+1], hit_list[t+2]
        base = cuda.atomic.add(n_verts, 0, 3)
        if base + 3 > max_v:
            return
        for k, ee in enumerate((e0, e1, e2)):
            verts_out[base+k, 0] = epts[ee, 0] / N
            verts_out[base+k, 1] = epts[ee, 1] / N
            verts_out[base+k, 2] = epts[ee, 2] / N

@cuda.jit
def render_mesh(verts, n, out, W, H, iso_color):
    """简单点/三角栅格化投影。"""
    i = cuda.grid(1)
    if i >= n // 3:
        return
    # 三角 i
    for k in range(3):
        p = verts[i*3+k]
        # 投影
        z = p[2] + 1.5
        if z < 0.1:
            continue
        px = int((p[0]-0.5)/z*2*W + W/2)
        py = int(-(p[1]-0.5)/z*2*H + H/2)
        if 0 <= px < W and 0 <= py < H:
            out[py, px, 0] = 200
            out[py, px, 1] = 180
            out[py, px, 2] = 160

def make_sphere_field(N=32):
    vol = np.zeros((N,N,N), np.float32)
    c = N/2
    for z in range(N):
        for y in range(N):
            for x in range(N):
                r = math.sqrt((x-c)**2+(y-c)**2+(z-c)**2) / (N*0.35)
                vol[z,y,x] = max(0, 1 - r)
    return vol

def main():
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    print("=" * 60)
    print("039 | Marching Cubes 等值面")
    print("=" * 60)
    N = 32
    vol = make_sphere_field(N)
    max_v = 200000
    d_vol = cuda.to_device(vol)
    d_verts = cuda.device_array((max_v, 3), np.float32)
    d_nv = cuda.to_device(np.zeros(1, np.int32))
    block = (4,4,4)
    grid = ((N+3)//4, (N+3)//4, (N+3)//4)
    iso = 0.5
    mc_extract[grid, block](d_vol, N, np.float32(iso), d_verts, d_nv, max_v)
    cuda.synchronize()
    nv = int(d_nv.copy_to_host()[0])
    print(f"等值面 iso={iso}: 顶点数={nv}")
    W, H = 400, 400
    d_img = cuda.to_device(np.zeros((H, W, 3), np.uint8))
    if nv >= 3:
        render_mesh[(nv//3+255)//256, 256](d_verts, nv, d_img, W, H, 0)
    img = d_img.copy_to_host()
    Image.fromarray(img).save(out / "mc_iso.png")
    # 另一等值
    d_nv = cuda.to_device(np.zeros(1, np.int32))
    mc_extract[grid, block](d_vol, N, np.float32(0.3), d_verts, d_nv, max_v)
    print(f"等值面 iso=0.3: 顶点数={int(d_nv.copy_to_host()[0])}")
    print("✓ Marching Cubes 完成。")

if __name__ == "__main__":
    main()
