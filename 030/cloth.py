#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""030 | 布料：Verlet + 距离约束 + 风力 + 法线渲染。"""
from pathlib import Path
import numpy as np
from numba import cuda
from PIL import Image

RES = 32
N = RES * RES

@cuda.jit
def verlet_integrate(pos, prev, n, dt, gravity, wind_x, wind_z, pinned):
    i = cuda.grid(1)
    if i >= n or pinned[i]:
        return
    x, y, z = pos[i, 0], pos[i, 1], pos[i, 2]
    px, py, pz = prev[i, 0], prev[i, 1], prev[i, 2]
    vx, vy, vz = x - px, y - py, z - pz
    nx = x + vx + wind_x * dt * dt
    ny = y + vy + gravity * dt * dt
    nz = z + vz + wind_z * dt * dt
    prev[i, 0], prev[i, 1], prev[i, 2] = x, y, z
    pos[i, 0], pos[i, 1], pos[i, 2] = nx, ny, nz
    if pos[i, 1] < 0:
        pos[i, 1] = 0
        prev[i, 1] = 0

@cuda.jit
def satisfy_constraints(pos, n, res, rest, pinned, iters):
    i = cuda.grid(1)
    if i >= n or pinned[i]:
        return
    r = res
    x = i % r
    y = i // r
    for _it in range(iters):
        if x < r - 1:
            j = i + 1
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            dist = (dx*dx+dy*dy+dz*dz)**0.5 + 1e-8
            diff = (dist - rest) / dist * 0.5
            if not pinned[i]:
                pos[i, 0] += dx * diff
                pos[i, 1] += dy * diff
                pos[i, 2] += dz * diff
            if not pinned[j]:
                pos[j, 0] -= dx * diff
                pos[j, 1] -= dy * diff
                pos[j, 2] -= dz * diff
        if y < r - 1:
            j = i + r
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            dist = (dx*dx+dy*dy+dz*dz)**0.5 + 1e-8
            diff = (dist - rest) / dist * 0.5
            if not pinned[i]:
                pos[i, 0] += dx * diff
                pos[i, 1] += dy * diff
                pos[i, 2] += dz * diff
            if not pinned[j]:
                pos[j, 0] -= dx * diff
                pos[j, 1] -= dy * diff
                pos[j, 2] -= dz * diff

@cuda.jit
def compute_normals(pos, nrm, res):
    i = cuda.grid(1)
    r = res
    if i >= r * r:
        return
    x = i % r
    y = i // r
    p0, p1, p2 = pos[i, 0], pos[i, 1], pos[i, 2]
    jx = i + (1 if x < r - 1 else 0)
    jy = i + (r if y < r - 1 else 0)
    dx = pos[jx, 0] - p0
    dy = pos[jx, 1] - p1
    dz = pos[jx, 2] - p2
    ex = pos[jy, 0] - p0
    ey = pos[jy, 1] - p1
    ez = pos[jy, 2] - p2
    nx = dy * ez - dz * ey
    ny = dz * ex - dx * ez
    nz = dx * ey - dy * ex
    L = (nx*nx+ny*ny+nz*nz)**0.5 + 1e-8
    nrm[i, 0] = nx / L
    nrm[i, 1] = ny / L
    nrm[i, 2] = nz / L

def render_cloth(pos, nrm, res, path):
    W, H = 400, 400
    img = np.zeros((H, W, 3), np.uint8)
    zbuf = np.full((H, W), 1e30)
    for y in range(res - 1):
        for x in range(res - 1):
            ids = [y*res+x, y*res+x+1, (y+1)*res+x]
            pts = pos[ids]
            ns = nrm[ids].mean(axis=0)
            nn = np.linalg.norm(ns) + 1e-8
            ns = ns / nn
            pavg = pts.mean(axis=0)
            z = pavg[2] + 0.5
            if z < 0.1:
                continue
            px = int((pavg[0] - 0.5) / z * W + W / 2)
            py = int((-pavg[1] + 0.3) / z * H + H / 2)
            if 0 <= px < W and 0 <= py < H and z < zbuf[py, px]:
                zbuf[py, px] = z
                ndl = max(0.0, float(np.dot(ns, np.array([0.3, 0.8, 0.4]))))
                c = np.array([0.2, 0.4, 0.85]) * (0.2 + 0.8 * ndl)
                img[py, px] = (np.clip(c, 0, 1) * 255).astype(np.uint8)
                # 填充邻域
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        xx, yy = px + dx, py + dy
                        if 0 <= xx < W and 0 <= yy < H:
                            img[yy, xx] = img[py, px]
    Image.fromarray(img).save(path)

def main():
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    print("=" * 60)
    print(f"030 | 布料 {RES}x{RES}")
    print("=" * 60)
    pos = np.zeros((N, 3), np.float32)
    for y in range(RES):
        for x in range(RES):
            pos[y * RES + x] = [x / (RES - 1), 1.0, y / (RES - 1) * 0.5]
    prev = pos.copy()
    pinned = np.zeros(N, np.bool_)
    for x in range(RES):
        pinned[x] = True
    rest = 1.0 / (RES - 1)
    d_pos = cuda.to_device(pos)
    d_prev = cuda.to_device(prev)
    d_pin = cuda.to_device(pinned)
    d_nrm = cuda.device_array((N, 3), np.float32)
    thr, blk = 128, (N + 127) // 128
    for step in range(80):
        verlet_integrate[blk, thr](d_pos, d_prev, N, np.float32(1/60), np.float32(-9.8),
                                   np.float32(1.5), np.float32(0.5), d_pin)
        satisfy_constraints[blk, thr](d_pos, N, RES, np.float32(rest), d_pin, 6)
    compute_normals[blk, thr](d_pos, d_nrm, RES)
    cuda.synchronize()
    render_cloth(d_pos.copy_to_host(), d_nrm.copy_to_host(), RES, out / "cloth.png")
    print("✓ 布料 Verlet+约束 完成。")

if __name__ == "__main__":
    main()
