#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""037 | 体渲染：程序化 CT + 光线步进 + 传递函数 + early termination。"""
from pathlib import Path
import numpy as np
from numba import cuda
from PIL import Image
import math, time

@cuda.jit(device=True)
def sample_vol(vol, x, y, z, N):
    if x < 0 or y < 0 or z < 0 or x > 1 or y > 1 or z > 1:
        return 0.0
    fx, fy, fz = x*(N-1), y*(N-1), z*(N-1)
    x0, y0, z0 = int(fx), int(fy), int(fz)
    x1 = x0+1 if x0+1 < N else N-1
    y1 = y0+1 if y0+1 < N else N-1
    z1 = z0+1 if z0+1 < N else N-1
    xd, yd, zd = fx-x0, fy-y0, fz-z0
    c000=vol[z0,y0,x0]; c100=vol[z0,y0,x1]
    c010=vol[z0,y1,x0]; c110=vol[z0,y1,x1]
    c001=vol[z1,y0,x0]; c101=vol[z1,y0,x1]
    c011=vol[z1,y1,x0]; c111=vol[z1,y1,x1]
    c00=c000*(1-xd)+c100*xd; c01=c001*(1-xd)+c101*xd
    c10=c010*(1-xd)+c110*xd; c11=c011*(1-xd)+c111*xd
    c0=c00*(1-yd)+c10*yd; c1=c01*(1-yd)+c11*yd
    return c0*(1-zd)+c1*zd

@cuda.jit(device=True)
def transfer(v, rgba):
    if v < 0.15:
        rgba[0]=rgba[1]=rgba[2]=rgba[3]=0.0
    elif v < 0.35:
        t=(v-0.15)/0.2
        rgba[0]=0.9*t; rgba[1]=0.6*t; rgba[2]=0.5*t; rgba[3]=0.05*t
    elif v < 0.55:
        t=(v-0.35)/0.2
        rgba[0]=0.8; rgba[1]=0.3+0.2*t; rgba[2]=0.2; rgba[3]=0.15
    else:
        t=(v-0.55)/0.3
        if t > 1.0: t = 1.0
        rgba[0]=1.0; rgba[1]=1.0; rgba[2]=0.95; rgba[3]=0.4*t+0.2

@cuda.jit
def volume_raymarch(vol, out, W, H, N, rot_y):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= W or y >= H: return
    u = (x+0.5)/W*2-1; v = 1-(y+0.5)/H*2
    cy, sy = math.cos(rot_y), math.sin(rot_y)
    ox, oy, oz = 0.0, 0.0, -1.8
    dx, dy, dz = u*0.7, v*0.7, 1.0
    inv = 1.0/math.sqrt(dx*dx+dy*dy+dz*dz)
    dx*=inv; dy*=inv; dz*=inv
    acc_r=acc_g=acc_b=0.0; T=1.0
    rgba = cuda.local.array(4, np.float32)
    for s in range(160):
        t = 0.8 + s * 0.015
        px, py, pz = ox+dx*t, oy+dy*t, oz+dz*t
        rx = px*cy + pz*sy
        ry = py
        rz = -px*sy + pz*cy
        dens = sample_vol(vol, rx*0.4+0.5, ry*0.4+0.5, rz*0.4+0.5, N)
        transfer(dens, rgba)
        a = rgba[3]
        if a > 1e-4:
            acc_r += T*a*rgba[0]
            acc_g += T*a*rgba[1]
            acc_b += T*a*rgba[2]
            T *= (1.0 - a)
            if T < 0.01: break
    acc_r += T*0.05; acc_g += T*0.05; acc_b += T*0.08
    if acc_r > 1: acc_r=1
    if acc_g > 1: acc_g=1
    if acc_b > 1: acc_b=1
    out[y,x,0]=acc_r*255; out[y,x,1]=acc_g*255; out[y,x,2]=acc_b*255

def make_ct_volume(N=48):
    vol = np.zeros((N,N,N), np.float32)
    c = N//2
    for z in range(N):
        for y in range(N):
            for x in range(N):
                dx,dy,dz = (x-c)/c,(y-c)/c,(z-c)/c
                r = math.sqrt(dx*dx+dy*dy+dz*dz)
                if 0.7 < r < 0.85: vol[z,y,x]=0.9
                elif r < 0.7:
                    vol[z,y,x]=0.25+0.15*math.sin(x*0.3)*math.cos(y*0.3)
                    if abs(dx)<0.15 and abs(dy)<0.1 and dz>-0.2: vol[z,y,x]=0.7
    return vol

def main():
    out=Path(__file__).parent/"output"; out.mkdir(exist_ok=True)
    print("="*60); print("037 | 体渲染"); print("="*60)
    N=48; vol=make_ct_volume(N)
    d_vol=cuda.to_device(vol)
    W,H=400,300
    d_out=cuda.device_array((H,W,3), np.float32)
    t0=time.perf_counter()
    volume_raymarch[((W+15)//16,(H+15)//16),(16,16)](d_vol,d_out,W,H,N,0.6)
    cuda.synchronize()
    print(f"渲染: {(time.perf_counter()-t0)*1000:.1f} ms")
    Image.fromarray(np.clip(d_out.copy_to_host(),0,255).astype(np.uint8)).save(out/"volume.png")
    print("✓ CT 体渲染完成。")

if __name__=="__main__":
    main()
