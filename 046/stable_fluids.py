#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""046 | Jos Stam 稳定流体：advect + diffuse + project；烟密度渲染。"""
from pathlib import Path
import numpy as np
from numba import cuda, float32
from PIL import Image
import time

N = 128

@cuda.jit
def add_source(x, s, dt, n):
    i = cuda.grid(1)
    if i < n*n:
        x[i] += dt * s[i]

@cuda.jit
def set_bnd(b, x, n):
    """边界条件。"""
    i = cuda.grid(1)
    if i >= n:
        return
    # 简化：四边
    if b == 1:
        x[i] = -x[n+i]  # 左——用扁平索引需小心
    # 用 2D 核更清晰

@cuda.jit
def advect(b, d, d0, u, v, dt, n):
    """半拉格朗日平流。d/u/v 为 n*n 扁平或 2D。"""
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if i < 1 or j < 1 or i >= n-1 or j >= n-1:
        return
    dt0 = dt * (n - 2)
    x = i - dt0 * u[j, i]
    y = j - dt0 * v[j, i]
    if x < 0.5: x = 0.5
    if x > n-1.5: x = n-1.5
    if y < 0.5: y = 0.5
    if y > n-1.5: y = n-1.5
    i0 = int(x); i1 = i0+1
    j0 = int(y); j1 = j0+1
    s1 = x-i0; s0 = 1-s1
    t1 = y-j0; t0 = 1-t1
    d[j, i] = s0*(t0*d0[j0,i0]+t1*d0[j1,i0]) + s1*(t0*d0[j0,i1]+t1*d0[j1,i1])

@cuda.jit
def diffuse(b, x, x0, diff, dt, n, iter_n):
    """Gauss-Seidel 扩散（多次迭代）。"""
    a = dt * diff * (n-2) * (n-2)
    for k in range(iter_n):
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        if 1 <= i < n-1 and 1 <= j < n-1:
            x[j,i] = (x0[j,i] + a*(x[j,i-1]+x[j,i+1]+x[j-1,i]+x[j+1,i])) / (1+4*a)
        cuda.syncthreads()

@cuda.jit
def project_div(u, v, p, div, n):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if 1 <= i < n-1 and 1 <= j < n-1:
        div[j,i] = -0.5*(u[j,i+1]-u[j,i-1]+v[j+1,i]-v[j-1,i]) / n
        p[j,i] = 0.0

@cuda.jit
def project_gs(p, div, n):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if 1 <= i < n-1 and 1 <= j < n-1:
        p[j,i] = (div[j,i] + p[j,i-1]+p[j,i+1]+p[j-1,i]+p[j+1,i]) / 4.0

@cuda.jit
def project_apply(u, v, p, n):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if 1 <= i < n-1 and 1 <= j < n-1:
        u[j,i] -= 0.5 * n * (p[j,i+1]-p[j,i-1])
        v[j,i] -= 0.5 * n * (p[j+1,i]-p[j-1,i])

@cuda.jit
def inject(dens, u, v, n, cx, cy, frame):
    """注入烟与速度。"""
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if i >= n or j >= n:
        return
    dx, dy = i-cx, j-cy
    if dx*dx+dy*dy < 20:
        dens[j,i] += 100.0
        u[j,i] += 2.0 * math.sin(frame*0.1)
        v[j,i] -= 3.0  # 向上（注意 j 向下则用负）

import math

def step_fluid(d_dens, d_dens0, d_u, d_v, d_u0, d_v0, d_p, d_div, n, dt, visc, diff, frame):
    grid = ((n+15)//16, (n+15)//16); block=(16,16)
    inject[grid,block](d_dens, d_u, d_v, n, n//2, n-10, frame)
    # 速度扩散+平流+投影
    cuda.to_device(d_u.copy_to_host() if False else 0)  # no-op placeholder
    # 简化管线
    d_u0.copy_from_device if False else None
    # 用 host 交换指针风格：拷贝
    # dens: diffuse + advect
    d_dens0[:] = d_dens  # device array slice assign may not work
    # 直接调用
    diffuse[grid,block](0, d_dens0, d_dens, diff, dt, n, 4)
    advect[grid,block](0, d_dens, d_dens0, d_u, d_v, dt, n)
    diffuse[grid,block](1, d_u0, d_u, visc, dt, n, 4)
    diffuse[grid,block](2, d_v0, d_v, visc, dt, n, 4)
    # 交换：u0 是扩散结果，再 advect
    advect[grid,block](1, d_u, d_u0, d_u0, d_v0, dt, n)
    advect[grid,block](2, d_v, d_v0, d_u0, d_v0, dt, n)
    project_div[grid,block](d_u, d_v, d_p, d_div, n)
    for _ in range(10):
        project_gs[grid,block](d_p, d_div, n)
    project_apply[grid,block](d_u, d_v, d_p, n)

def main():
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    print("=" * 60)
    print(f"046 | 稳定流体 {N}x{N}")
    print("=" * 60)
    dens = cuda.to_device(np.zeros((N,N), np.float32))
    dens0 = cuda.to_device(np.zeros((N,N), np.float32))
    u = cuda.to_device(np.zeros((N,N), np.float32))
    v = cuda.to_device(np.zeros((N,N), np.float32))
    u0 = cuda.to_device(np.zeros((N,N), np.float32))
    v0 = cuda.to_device(np.zeros((N,N), np.float32))
    p = cuda.to_device(np.zeros((N,N), np.float32))
    div = cuda.to_device(np.zeros((N,N), np.float32))
    grid = ((N+15)//16, (N+15)//16); block=(16,16)
    dt, visc, diff = 0.1, 0.0001, 0.0001
    t0 = time.perf_counter()
    for frame in range(60):
        inject[grid,block](dens, u, v, N, N//2, N-8, frame)
        # dens step
        dens0.copy_to_device(dens.copy_to_host())  # 简化交换
        # 手动：先把 dens 拷到 dens0
        tmp = dens.copy_to_host(); dens0 = cuda.to_device(tmp)
        diffuse[grid,block](0, dens, dens0, np.float32(diff), np.float32(dt), N, 8)
        tmp = dens.copy_to_host(); dens0 = cuda.to_device(tmp)
        advect[grid,block](0, dens, dens0, u, v, np.float32(dt), N)
        # velocity
        tmpu = u.copy_to_host(); u0 = cuda.to_device(tmpu)
        tmpv = v.copy_to_host(); v0 = cuda.to_device(tmpv)
        diffuse[grid,block](1, u, u0, np.float32(visc), np.float32(dt), N, 4)
        diffuse[grid,block](2, v, v0, np.float32(visc), np.float32(dt), N, 4)
        tmpu = u.copy_to_host(); u0 = cuda.to_device(tmpu)
        tmpv = v.copy_to_host(); v0 = cuda.to_device(tmpv)
        advect[grid,block](1, u, u0, u0, v0, np.float32(dt), N)
        advect[grid,block](2, v, v0, u0, v0, np.float32(dt), N)
        project_div[grid,block](u, v, p, div, N)
        for _ in range(12):
            project_gs[grid,block](p, div, N)
        project_apply[grid,block](u, v, p, N)
    cuda.synchronize()
    print(f"60 步: {(time.perf_counter()-t0)*1000:.1f} ms")
    d = dens.copy_to_host()
    d = d / (d.max()+1e-6)
    rgb = np.zeros((N,N,3), np.uint8)
    rgb[:,:,0] = (d*255).astype(np.uint8)
    rgb[:,:,1] = (d*120).astype(np.uint8)
    rgb[:,:,2] = (d*40).astype(np.uint8)
    Image.fromarray(rgb).save(out/"smoke.png")
    # 速度场可视化
    uh, vh = u.copy_to_host(), v.copy_to_host()
    mag = np.sqrt(uh**2+vh**2)
    mag = (mag/(mag.max()+1e-6)*255).astype(np.uint8)
    Image.fromarray(mag).save(out/"velocity.png")
    print("✓ Navier-Stokes 稳定流体完成。")

if __name__ == "__main__":
    main()
