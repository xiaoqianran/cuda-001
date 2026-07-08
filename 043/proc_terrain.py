#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""043 | 程序化地形：fBM 高度 + 生物群系 + chunk + 渲染。"""
from pathlib import Path
import numpy as np
from numba import cuda
from PIL import Image
import math

@cuda.jit(device=True)
def hash2(x, z):
    n = (x * 374761393 + z * 668265263) & 0x7fffffff
    return (n % 10000) / 10000.0

@cuda.jit(device=True)
def noise(x, z):
    xi, zi = int(math.floor(x)), int(math.floor(z))
    xf, zf = x - xi, z - zi
    u = xf*xf*(3-2*xf); v = zf*zf*(3-2*zf)
    a,b,c,d = hash2(xi,zi), hash2(xi+1,zi), hash2(xi,zi+1), hash2(xi+1,zi+1)
    return a+(b-a)*u+(c-a)*v+(a-b-c+d)*u*v

@cuda.jit(device=True)
def fbm(x, z, octaves):
    a=0.0; f=1.0; amp=1.0; nrm=0.0
    for i in range(octaves):
        a += noise(x*f, z*f)*amp; nrm += amp; amp *= 0.5; f *= 2.0
    return a/nrm

@cuda.jit
def gen_chunk_height(height, biome, cx, cz, size, scale):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    z = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= size or z >= size: return
    wx = (cx * size + x) * scale
    wz = (cz * size + z) * scale
    h = fbm(wx, wz, 6) * 80.0
    temp = fbm(wx*0.3+10, wz*0.3, 3)
    humid = fbm(wx*0.3-5, wz*0.3+3, 3)
    if h < 5: b = 0
    elif h < 12 and humid < 0.3: b = 1
    elif h > 55: b = 4
    elif humid > 0.55 and h < 40: b = 3
    else: b = 2
    height[z, x] = h
    biome[z, x] = b

@cuda.jit
def render_terrain(height, biome, size, out, W, H, cam_x, cam_y, cam_z, yaw):
    px = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    py = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if px >= W or py >= H: return
    u = (px+0.5)/W*2-1; v = 1-(py+0.5)/H*2
    cy, sy = math.cos(yaw), math.sin(yaw)
    dx, dy, dz = u*0.8, v*0.6-0.2, 1.0
    rdx = dx*cy + dz*sy
    rdz = -dx*sy + dz*cy
    inv = 1.0/math.sqrt(rdx*rdx+dy*dy+rdz*rdz)
    rdx*=inv; dy*=inv; rdz*=inv
    col_r, col_g, col_b = 0.5, 0.7, 0.95
    t = 1.0
    for s in range(180):
        x = cam_x + rdx * t
        y = cam_y + dy * t
        z = cam_z + rdz * t
        fx = (x / 2.0) % size
        fz = (z / 2.0) % size
        if fx < 0: fx += size
        if fz < 0: fz += size
        ix, iz = int(fx) % size, int(fz) % size
        h = height[iz, ix]
        if y < h:
            b = biome[iz, ix]
            if b == 0: col_r,col_g,col_b = 0.1,0.2,0.6
            elif b == 1: col_r,col_g,col_b = 0.85,0.78,0.5
            elif b == 2: col_r,col_g,col_b = 0.25,0.55,0.15
            elif b == 3: col_r,col_g,col_b = 0.1,0.35,0.1
            else: col_r,col_g,col_b = 0.9,0.92,0.95
            col_r *= 0.7; col_g *= 0.7; col_b *= 0.7
            break
        step = (y - h) * 0.3
        if step < 0.5: step = 0.5
        t += step
        if t > 300: break
    fog = 1 - math.exp(-t*0.008)
    col_r = col_r*(1-fog)+0.6*fog
    col_g = col_g*(1-fog)+0.7*fog
    col_b = col_b*(1-fog)+0.85*fog
    if col_r>1: col_r=1
    if col_g>1: col_g=1
    if col_b>1: col_b=1
    out[py,px,0]=col_r*255; out[py,px,1]=col_g*255; out[py,px,2]=col_b*255

def main():
    out=Path(__file__).parent/"output"; out.mkdir(exist_ok=True)
    print("="*60); print("043 | 程序化地形 + 生物群系"); print("="*60)
    size=128
    d_h=cuda.device_array((size,size), np.float32)
    d_b=cuda.device_array((size,size), np.int32)
    grid_c=((size+15)//16,(size+15)//16)
    for cx,cz in [(0,0),(1,0),(0,1)]:
        gen_chunk_height[grid_c,(16,16)](d_h,d_b,cx,cz,size,0.05)
        print(f"  chunk ({cx},{cz}) 已生成")
    W,H=640,360
    d_out=cuda.device_array((H,W,3), np.float32)
    render_terrain[((W+15)//16,(H+15)//16),(16,16)](d_h,d_b,size,d_out,W,H,0.0,40.0,0.0,0.5)
    cuda.synchronize()
    Image.fromarray(np.clip(d_out.copy_to_host(),0,255).astype(np.uint8)).save(out/"terrain.png")
    print("✓ 程序化地形完成。")

if __name__=="__main__":
    main()
