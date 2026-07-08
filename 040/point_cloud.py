#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""040 | 点云：程序化点 + LOD + 视锥剔除 + 高度着色。"""
from pathlib import Path
import numpy as np
from numba import cuda
from PIL import Image
import time

N = 150_000

@cuda.jit
def frustum_lod_render(pts, cols, out, zbuf, n, W, H, eye_z, lod_factor):
    i = cuda.grid(1)
    if i >= n: return
    x, y, z = pts[i,0], pts[i,1], pts[i,2]
    zc = eye_z - z
    if zc < 0.5: return
    stride = 1 + int(zc * lod_factor)
    if (i % stride) != 0: return
    px = int((x / zc) * W * 0.5 + W * 0.5)
    py = int((-y / zc) * H * 0.5 + H * 0.5)
    if px < 0 or py < 0 or px >= W or py >= H: return
    size = int(3.0 / zc * 2)
    if size < 1: size = 1
    if size > 4: size = 4
    for dy in range(-size, size+1):
        for dx in range(-size, size+1):
            xx, yy = px+dx, py+dy
            if xx >= 0 and yy >= 0 and xx < W and yy < H:
                old = cuda.atomic.min(zbuf, (yy, xx), zc)
                if zc <= old + 1e-3:
                    out[yy, xx, 0] = cols[i, 0]
                    out[yy, xx, 1] = cols[i, 1]
                    out[yy, xx, 2] = cols[i, 2]

def main():
    out = Path(__file__).parent/"output"; out.mkdir(exist_ok=True)
    print("="*60); print(f"040 | 点云渲染 N={N}"); print("="*60)
    rng = np.random.default_rng(0)
    pts = rng.uniform(-5, 5, (N, 3)).astype(np.float32)
    pts[:, 1] = rng.uniform(0, 0.2, N)
    for b in range(20):
        idx = slice(b*1000, (b+1)*1000)
        cx, cz = rng.uniform(-4,4), rng.uniform(-4,4)
        pts[idx,0] = cx + rng.uniform(-0.3,0.3,1000)
        pts[idx,2] = cz + rng.uniform(-0.3,0.3,1000)
        pts[idx,1] = rng.uniform(0, 2.5, 1000)
    cols = np.zeros((N,3), np.uint8)
    h = pts[:,1]
    cols[:,0] = np.clip(h/2.5*255, 0, 255).astype(np.uint8)
    cols[:,1] = np.clip((1-h/2.5)*200, 0, 255).astype(np.uint8)
    cols[:,2] = 80
    d_pts = cuda.to_device(pts); d_cols = cuda.to_device(cols)
    W, H = 640, 360
    d_out = cuda.to_device(np.zeros((H,W,3), np.uint8))
    d_z = cuda.to_device(np.full((H,W), 1e30, np.float32))
    thr, blk = 256, (N+255)//256
    t0 = time.perf_counter()
    frustum_lod_render[blk, thr](d_pts, d_cols, d_out, d_z, N, W, H, 8.0, 0.15)
    cuda.synchronize()
    print(f"渲染: {(time.perf_counter()-t0)*1000:.1f} ms")
    Image.fromarray(d_out.copy_to_host()).save(out/"pointcloud.png")
    print("✓ 点云 LOD + 视锥剔除完成。")

if __name__=="__main__":
    main()
