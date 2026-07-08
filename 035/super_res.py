#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""035 | 传统 GPU 超分：双三次 + Lanczos；PSNR/SSIM。"""
from pathlib import Path
import numpy as np
from numba import cuda, uint8
from PIL import Image
import math, time, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.image_utils import make_demo_rgb, save_png

@cuda.jit(device=True)
def cubic(x):
    ax = abs(x)
    if ax <= 1.0:
        return 1.5*ax*ax*ax - 2.5*ax*ax + 1.0
    if ax < 2.0:
        return -0.5*ax*ax*ax + 2.5*ax*ax - 4.0*ax + 2.0
    return 0.0

@cuda.jit(device=True)
def lanczos_w(t, a):
    if t == 0.0:
        return 1.0
    if abs(t) >= a:
        return 0.0
    pix = 3.141592653589793
    return a * math.sin(pix * t) * math.sin(pix * t / a) / (pix * pix * t * t)

@cuda.jit(device=True)
def ci(v, lo, hi):
    if v < lo: return lo
    if v > hi: return hi
    return v

@cuda.jit
def bicubic_kernel(src, dst, sw, sh, dw, dh):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= dw or y >= dh: return
    sx = (x + 0.5) * sw / dw - 0.5
    sy = (y + 0.5) * sh / dh - 0.5
    ix = int(math.floor(sx)); iy = int(math.floor(sy))
    for c in range(3):
        acc = 0.0; wsum = 0.0
        for j in range(-1, 3):
            for i in range(-1, 3):
                w = cubic(sx-(ix+i)) * cubic(sy-(iy+j))
                xx = ci(ix+i, 0, sw-1); yy = ci(iy+j, 0, sh-1)
                acc += w * src[yy, xx, c]; wsum += w
        v = acc/wsum if wsum > 0 else 0.0
        if v < 0.0: v = 0.0
        if v > 255.0: v = 255.0
        dst[y, x, c] = uint8(v)

@cuda.jit
def lanczos_kernel(src, dst, sw, sh, dw, dh, a):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= dw or y >= dh: return
    sx = (x + 0.5) * sw / dw - 0.5
    sy = (y + 0.5) * sh / dh - 0.5
    ix = int(math.floor(sx)); iy = int(math.floor(sy))
    aa = int(a)
    for c in range(3):
        acc = 0.0; wsum = 0.0
        for j in range(-aa+1, aa+1):
            for i in range(-aa+1, aa+1):
                w = lanczos_w(sx-(ix+i), a) * lanczos_w(sy-(iy+j), a)
                xx = ci(ix+i, 0, sw-1); yy = ci(iy+j, 0, sh-1)
                acc += w * src[yy, xx, c]; wsum += w
        v = acc/wsum if wsum else 0.0
        if v < 0.0: v = 0.0
        if v > 255.0: v = 255.0
        dst[y, x, c] = uint8(v)

@cuda.jit
def sharpen_kernel(src, dst, W, H, amount):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= W or y >= H: return
    for c in range(3):
        acc = 0.0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                acc += src[ci(y+dy,0,H-1), ci(x+dx,0,W-1), c]
        blur = acc / 9.0
        v = src[y,x,c] + amount * (src[y,x,c] - blur)
        if v < 0.0: v = 0.0
        if v > 255.0: v = 255.0
        dst[y,x,c] = uint8(v)

def psnr(a, b):
    mse = np.mean((a.astype(np.float64)-b.astype(np.float64))**2)
    return 99.0 if mse < 1e-10 else 10*math.log10(255**2/mse)

def ssim_simple(a, b):
    a=a.astype(np.float64); b=b.astype(np.float64)
    if a.ndim==3:
        a=0.299*a[:,:,0]+0.587*a[:,:,1]+0.114*a[:,:,2]
        b=0.299*b[:,:,0]+0.587*b[:,:,1]+0.114*b[:,:,2]
    mu_a, mu_b = a.mean(), b.mean()
    sa, sb = a.var(), b.var()
    sab = ((a-mu_a)*(b-mu_b)).mean()
    c1,c2=(0.01*255)**2,(0.03*255)**2
    return ((2*mu_a*mu_b+c1)*(2*sab+c2))/((mu_a**2+mu_b**2+c1)*(sa+sb+c2))

def main():
    out = Path(__file__).parent/"output"; out.mkdir(exist_ok=True)
    print("="*60); print("035 | 传统 GPU 超分辨率"); print("="*60)
    gt = make_demo_rgb(640, 360)
    lr = np.array(Image.fromarray(gt).resize((320, 180), Image.BILINEAR))
    dh, dw = lr.shape[0]*2, lr.shape[1]*2
    d_lr = cuda.to_device(np.ascontiguousarray(lr))
    d_bc = cuda.device_array((dh,dw,3), np.uint8)
    d_lc = cuda.device_array((dh,dw,3), np.uint8)
    d_sh = cuda.device_array((dh,dw,3), np.uint8)
    grid = ((dw+15)//16, (dh+15)//16)
    t0=time.perf_counter()
    bicubic_kernel[grid,(16,16)](d_lr,d_bc,lr.shape[1],lr.shape[0],dw,dh)
    sharpen_kernel[grid,(16,16)](d_bc,d_sh,dw,dh,0.6)
    cuda.synchronize(); t_bc=(time.perf_counter()-t0)*1000
    t0=time.perf_counter()
    lanczos_kernel[grid,(16,16)](d_lr,d_lc,lr.shape[1],lr.shape[0],dw,dh,3.0)
    cuda.synchronize(); t_lc=(time.perf_counter()-t0)*1000
    bc, lc = d_sh.copy_to_host(), d_lc.copy_to_host()
    gt_r = np.array(Image.fromarray(gt).resize((dw,dh), Image.BILINEAR))
    print(f"双三次+锐化: {t_bc:.2f} ms | PSNR={psnr(bc,gt_r):.2f} SSIM={ssim_simple(bc,gt_r):.4f}")
    print(f"Lanczos:      {t_lc:.2f} ms | PSNR={psnr(lc,gt_r):.2f} SSIM={ssim_simple(lc,gt_r):.4f}")
    save_png(out/"lr.png", lr); save_png(out/"bicubic.png", bc); save_png(out/"lanczos.png", lc)
    print("✓ 超分 + 客观指标完成。")

if __name__ == "__main__":
    main()
