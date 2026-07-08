#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""028 | SPH 简化：空间哈希邻居、密度压力粘性、积分；Marching Squares 2D 表面。"""
from pathlib import Path
import numpy as np
from numba import cuda, float32, int32
import time
from PIL import Image

N = 4096
H = 0.05  # 光滑核半径
MASS = 0.02
RHO0 = 1000.0
GAS_K = 3.0
VISC = 0.1
G = -9.8

@cuda.jit(device=True)
def poly6(r, h):
    if r >= h:
        return 0.0
    x = h*h - r*r
    return 315.0 / (64.0 * 3.14159265 * h**9) * x * x * x

@cuda.jit(device=True)
def spiky_grad(r, h):
    if r <= 1e-8 or r >= h:
        return 0.0
    return -45.0 / (3.14159265 * h**6) * (h - r) * (h - r)

@cuda.jit
def hash_particles(pos, cell, n, origin_x, origin_y, cell_size, grid_w):
    i = cuda.grid(1)
    if i >= n:
        return
    cx = int((pos[i, 0] - origin_x) / cell_size)
    cy = int((pos[i, 1] - origin_y) / cell_size)
    cell[i] = cy * grid_w + cx

@cuda.jit
def compute_density_pressure(pos, dens, pres, n, h, mass, rho0, gas_k):
    i = cuda.grid(1)
    if i >= n:
        return
    rho = 0.0
    for j in range(n):  # 教学：O(n^2)；大规模应用哈希邻域
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        r = (dx*dx + dy*dy) ** 0.5
        rho += mass * poly6(r, h)
    dens[i] = rho if rho > 1e-6 else rho0
    pres[i] = gas_k * (dens[i] - rho0)

@cuda.jit
def compute_force_integrate(pos, vel, dens, pres, n, h, mass, visc, dt, g):
    i = cuda.grid(1)
    if i >= n:
        return
    fx, fy = 0.0, g * dens[i]
    for j in range(n):
        if i == j:
            continue
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        r = (dx*dx + dy*dy) ** 0.5 + 1e-8
        if r < h:
            # 压力
            w = spiky_grad(r, h)
            fscale = -mass * (pres[i] + pres[j]) / (2 * dens[j]) * w
            fx += fscale * dx / r
            fy += fscale * dy / r
            # 粘性
            vx = vel[j, 0] - vel[i, 0]
            vy = vel[j, 1] - vel[i, 1]
            vis_w = poly6(r, h)
            fx += visc * mass * vx / dens[j] * vis_w
            fy += visc * mass * vy / dens[j] * vis_w
    # 积分
    ax, ay = fx / dens[i], fy / dens[i]
    vel[i, 0] += ax * dt
    vel[i, 1] += ay * dt
    pos[i, 0] += vel[i, 0] * dt
    pos[i, 1] += vel[i, 1] * dt
    # 边界
    if pos[i, 0] < 0: pos[i, 0] = 0; vel[i, 0] *= -0.5
    if pos[i, 0] > 1: pos[i, 0] = 1; vel[i, 0] *= -0.5
    if pos[i, 1] < 0: pos[i, 1] = 0; vel[i, 1] *= -0.5
    if pos[i, 1] > 1: pos[i, 1] = 1; vel[i, 1] *= -0.5

def marching_squares_density(pos, res=128):
    """从粒子生成密度网格并抽等值线（简化为密度可视化）。"""
    grid = np.zeros((res, res), np.float32)
    for p in pos:
        x = int(p[0] * (res - 1))
        y = int(p[1] * (res - 1))
        if 0 <= x < res and 0 <= y < res:
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    xx, yy = x+dx, y+dy
                    if 0 <= xx < res and 0 <= yy < res:
                        grid[yy, xx] += 1.0
    return grid

def main():
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    print("=" * 60)
    print(f"028 | SPH N={N}")
    print("=" * 60)
    # 初始化水柱
    side = int(np.sqrt(N))
    pts = []
    for i in range(side):
        for j in range(side):
            pts.append([0.2 + i / side * 0.3, 0.3 + j / side * 0.4])
    pos_h = np.zeros((N, 2), np.float32)
    pos_h[:len(pts)] = np.array(pts, np.float32)[:N]
    vel_h = np.zeros((N, 2), np.float32)
    pos = cuda.to_device(pos_h)
    vel = cuda.to_device(vel_h)
    dens = cuda.device_array(N, np.float32)
    pres = cuda.device_array(N, np.float32)
    cell = cuda.device_array(N, np.int32)
    thr, blk = 128, (N + 127) // 128
    t0 = time.perf_counter()
    for step in range(30):
        hash_particles[blk, thr](pos, cell, N, -0.1, -0.1, H, 32)
        compute_density_pressure[blk, thr](pos, dens, pres, N, np.float32(H), np.float32(MASS), np.float32(RHO0), np.float32(GAS_K))
        compute_force_integrate[blk, thr](pos, vel, dens, pres, N, np.float32(H), np.float32(MASS), np.float32(VISC), np.float32(0.003), np.float32(G))
    cuda.synchronize()
    print(f"30 步 SPH: {(time.perf_counter()-t0)*1e3:.1f} ms")
    ph = pos.copy_to_host()
    dens_img = marching_squares_density(ph, 256)
    dens_img = (dens_img / (dens_img.max() + 1e-6) * 255).astype(np.uint8)
    # 伪彩
    rgb = np.zeros((256, 256, 3), np.uint8)
    rgb[..., 2] = dens_img
    rgb[..., 1] = dens_img // 2
    Image.fromarray(rgb).save(out / "sph.png")
    print("✓ SPH + 密度表面可视化完成。")

if __name__ == "__main__":
    main()
