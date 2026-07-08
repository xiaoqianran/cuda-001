#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""038 | 流场：RK4 流线 + LIC + 纹理 advection。"""
from pathlib import Path
import numpy as np
from numba import cuda
from PIL import Image
import math

@cuda.jit(device=True)
def sample_field(field, x, y, W, H):
    ix = int(x); iy = int(y)
    if ix < 0: ix = 0
    if iy < 0: iy = 0
    if ix >= W: ix = W - 1
    if iy >= H: iy = H - 1
    return field[iy, ix, 0], field[iy, ix, 1]

@cuda.jit
def integrate_streamlines(field, lines, n_seeds, steps, dt, W, H):
    s = cuda.grid(1)
    if s >= n_seeds: return
    x = lines[s, 0, 0]; y = lines[s, 0, 1]
    for i in range(1, steps):
        k1x, k1y = sample_field(field, x, y, W, H)
        k2x, k2y = sample_field(field, x+0.5*dt*k1x, y+0.5*dt*k1y, W, H)
        k3x, k3y = sample_field(field, x+0.5*dt*k2x, y+0.5*dt*k2y, W, H)
        k4x, k4y = sample_field(field, x+dt*k3x, y+dt*k3y, W, H)
        x += dt/6*(k1x+2*k2x+2*k3x+k4x)
        y += dt/6*(k1y+2*k2y+2*k3y+k4y)
        if x < 0: x = 0.0
        if y < 0: y = 0.0
        if x > W-1: x = float(W-1)
        if y > H-1: y = float(H-1)
        lines[s, i, 0] = x; lines[s, i, 1] = y

@cuda.jit
def lic_kernel(field, noise, out, W, H, L):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= W or y >= H: return
    acc = 0.0; wsum = 0.0
    for direction in (-1, 1):
        px, py = float(x), float(y)
        for s in range(L):
            ix = int(px); iy = int(py)
            if ix < 0: ix = 0
            if iy < 0: iy = 0
            if ix >= W: ix = W-1
            if iy >= H: iy = H-1
            w = 1.0 - s / float(L)
            acc += noise[iy, ix] * w
            wsum += w
            fx, fy = sample_field(field, px, py, W, H)
            ln = math.sqrt(fx*fx+fy*fy) + 1e-6
            px += direction * fx / ln
            py += direction * fy / ln
    out[y, x] = acc / wsum

def make_field(W, H):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = W/2, H/2
    dx, dy = xx-cx, yy-cy
    field = np.zeros((H, W, 2), np.float32)
    field[..., 0] = -dy * 0.02 + 0.3
    field[..., 1] = dx * 0.02
    return field

def main():
    out = Path(__file__).parent/"output"; out.mkdir(exist_ok=True)
    print("="*60); print("038 | 流场可视化 RK4 + LIC"); print("="*60)
    W, H = 256, 256
    field = make_field(W, H)
    n_seeds, steps = 40, 80
    seeds = np.zeros((n_seeds, steps, 2), np.float32)
    for i in range(n_seeds):
        seeds[i,0,0] = (i%8+0.5)*W/8
        seeds[i,0,1] = (i//8+0.5)*H/5
    d_f = cuda.to_device(field)
    d_lines = cuda.to_device(seeds)
    integrate_streamlines[(n_seeds+31)//32, 32](d_f, d_lines, n_seeds, steps, 1.5, W, H)
    lines = d_lines.copy_to_host()
    img = np.zeros((H,W,3), np.uint8); img[:] = 20
    for s in range(n_seeds):
        for i in range(steps-1):
            x0,y0=int(lines[s,i,0]),int(lines[s,i,1])
            x1,y1=int(lines[s,i+1,0]),int(lines[s,i+1,1])
            for t in range(10):
                xx=int(x0+(x1-x0)*t/10); yy=int(y0+(y1-y0)*t/10)
                if 0<=xx<W and 0<=yy<H: img[yy,xx]=[100,200,255]
    Image.fromarray(img).save(out/"streamlines.png")
    noise = np.random.default_rng(0).random((H,W)).astype(np.float32)
    d_n = cuda.to_device(noise)
    d_lic = cuda.device_array((H,W), np.float32)
    lic_kernel[((W+15)//16,(H+15)//16),(16,16)](d_f, d_n, d_lic, W, H, 12)
    lic = d_lic.copy_to_host()
    lic_img = (lic/(lic.max()+1e-6)*255).astype(np.uint8)
    Image.fromarray(lic_img).save(out/"lic.png")
    adv = noise.copy()
    for y in range(H):
        for x in range(W):
            fx, fy = field[y,x]
            sx = int(x - fx*2); sy = int(y - fy*2)
            if sx < 0: sx = 0
            if sy < 0: sy = 0
            if sx >= W: sx = W-1
            if sy >= H: sy = H-1
            adv[y,x] = noise[sy,sx]
    Image.fromarray((adv*255).astype(np.uint8)).save(out/"advection.png")
    print("✓ 流线 / LIC / advection 完成。")

if __name__=="__main__":
    main()
