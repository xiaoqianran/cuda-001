#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""034 | 非 AI 风格化：Kuwahara 油画、素描、水彩。"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
from numba import cuda, float32, uint8
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.image_utils import make_demo_rgb, save_png

@cuda.jit(device=True)
def ci(v, lo, hi):
    if v < lo: return lo
    if v > hi: return hi
    return v

@cuda.jit
def kuwahara_kernel(src, dst, W, H, r):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= W or y >= H: return
    best_var = 1e30
    bm0 = bm1 = bm2 = 0.0
    for q in range(4):
        x0 = x + (-r if (q % 2) == 0 else 0)
        x1 = x + (0 if (q % 2) == 0 else r)
        y0 = y + (-r if q < 2 else 0)
        y1 = y + (0 if q < 2 else r)
        s0=s1=s2=s00=s11=s22=0.0
        cnt = 0
        yy = y0
        while yy <= y1:
            xx = x0
            while xx <= x1:
                ix = ci(xx, 0, W-1); iy = ci(yy, 0, H-1)
                v0=float(src[iy,ix,0]); v1=float(src[iy,ix,1]); v2=float(src[iy,ix,2])
                s0+=v0; s1+=v1; s2+=v2; s00+=v0*v0; s11+=v1*v1; s22+=v2*v2; cnt+=1
                xx += 1
            yy += 1
        inv = 1.0/cnt
        m0,m1,m2 = s0*inv, s1*inv, s2*inv
        var = (s00*inv-m0*m0)+(s11*inv-m1*m1)+(s22*inv-m2*m2)
        if var < best_var:
            best_var=var; bm0,bm1,bm2=m0,m1,m2
    dst[y,x,0]=uint8(bm0); dst[y,x,1]=uint8(bm1); dst[y,x,2]=uint8(bm2)

@cuda.jit
def sketch_kernel(src, dst, W, H):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= W or y >= H: return
    def gray(ix, iy):
        ix=ci(ix,0,W-1); iy=ci(iy,0,H-1)
        return 0.299*src[iy,ix,0]+0.587*src[iy,ix,1]+0.114*src[iy,ix,2]
    gx = -gray(x-1,y-1)+gray(x+1,y-1)-2*gray(x-1,y)+2*gray(x+1,y)-gray(x-1,y+1)+gray(x+1,y+1)
    gy = -gray(x-1,y-1)-2*gray(x,y-1)-gray(x+1,y-1)+gray(x-1,y+1)+2*gray(x,y+1)+gray(x+1,y+1)
    edge = (gx*gx+gy*gy)**0.5
    if edge > 255.0: edge = 255.0
    g = gray(x,y)
    period = 6
    dx = (x % period) - period//2
    dy = (y % period) - period//2
    dist = (dx*dx+dy*dy)**0.5
    thresh = (255.0-g)/255.0 * (period*0.6)
    tone = 40.0 if dist < thresh else 240.0
    v = tone * (1.0 - edge/255.0*0.85)
    if v < 0.0: v = 0.0
    if v > 255.0: v = 255.0
    dst[y,x,0]=dst[y,x,1]=dst[y,x,2]=uint8(v)

@cuda.jit
def watercolor_kernel(src, dst, W, H):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= W or y >= H: return
    s0=s1=s2=0.0
    for dy in range(-2,3):
        for dx in range(-2,3):
            ix=ci(x+dx,0,W-1); iy=ci(y+dy,0,H-1)
            s0+=src[iy,ix,0]; s1+=src[iy,ix,1]; s2+=src[iy,ix,2]
    s0/=25; s1/=25; s2/=25
    levels=6.0
    q0 = (int(s0/255.0*levels+0.5)/levels)*255
    q1 = (int(s1/255.0*levels+0.5)/levels)*255
    q2 = (int(s2/255.0*levels+0.5)/levels)*255
    paper = 0.92 + 0.08 * (((x*12+y*78)%100)/100.0)
    def g(ix,iy):
        ix=ci(ix,0,W-1); iy=ci(iy,0,H-1)
        return 0.299*src[iy,ix,0]+0.587*src[iy,ix,1]+0.114*src[iy,ix,2]
    edge = abs(g(x+1,y)-g(x-1,y))+abs(g(x,y+1)-g(x,y-1))
    dark = 1.0 - (edge/80.0 if edge<80 else 1.0)*0.35
    if dark < 0.0: dark=0.0
    for c,q in enumerate((q0,q1,q2)):
        vv = q*paper*dark
        if vv > 255.0: vv=255.0
        dst[y,x,c]=uint8(vv)

def run(kernel, img, *extra):
    h,w = img.shape[:2]
    d_src=cuda.to_device(np.ascontiguousarray(img))
    d_dst=cuda.device_array_like(img)
    grid=((w+15)//16,(h+15)//16)
    kernel[grid,(16,16)](d_src,d_dst,w,h,*extra)
    return d_dst.copy_to_host()

def main():
    out=Path(__file__).parent/"output"; out.mkdir(exist_ok=True)
    print("="*60); print("034 | 风格迁移"); print("="*60)
    img=make_demo_rgb(640,360)
    for name,fn in [("oil",lambda:run(kuwahara_kernel,img,4)),
                    ("sketch",lambda:run(sketch_kernel,img)),
                    ("watercolor",lambda:run(watercolor_kernel,img))]:
        fn(); cuda.synchronize()
        t0=time.perf_counter()
        for _ in range(5):
            result=fn(); cuda.synchronize()
        ms=(time.perf_counter()-t0)/5*1000
        print(f"  {name:12s}: {ms:7.3f} ms")
        save_png(out/f"{name}.png", result)
    print("✓ 三种艺术风格完成。")

if __name__=="__main__":
    main()
