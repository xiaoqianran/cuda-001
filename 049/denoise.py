#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""049 | 4spp 噪声 + 辅助缓冲联合双边降噪；TensorRT 接口说明。"""
from pathlib import Path
import numpy as np
from numba import cuda
from PIL import Image
import time, math

@cuda.jit(device=True)
def ci(v, lo, hi):
    if v < lo: return lo
    if v > hi: return hi
    return v

@cuda.jit
def joint_bilateral(noisy, albedo, normal, out, W, H, sigma_s, sigma_c, sigma_n):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= W or y >= H: return
    r = 4
    ac0=ac1=ac2=0.0; wsum=0.0
    a0,a1,a2 = albedo[y,x,0], albedo[y,x,1], albedo[y,x,2]
    n0,n1,n2 = normal[y,x,0], normal[y,x,1], normal[y,x,2]
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            ix = ci(x+dx, 0, W-1); iy = ci(y+dy, 0, H-1)
            ws = math.exp(-(dx*dx+dy*dy)/(2*sigma_s*sigma_s))
            da0=albedo[iy,ix,0]-a0; da1=albedo[iy,ix,1]-a1; da2=albedo[iy,ix,2]-a2
            wc = math.exp(-(da0*da0+da1*da1+da2*da2)/(2*sigma_c*sigma_c))
            dn0=normal[iy,ix,0]-n0; dn1=normal[iy,ix,1]-n1; dn2=normal[iy,ix,2]-n2
            wn = math.exp(-(dn0*dn0+dn1*dn1+dn2*dn2)/(2*sigma_n*sigma_n))
            w = ws*wc*wn
            ac0 += w*noisy[iy,ix,0]; ac1 += w*noisy[iy,ix,1]; ac2 += w*noisy[iy,ix,2]
            wsum += w
    out[y,x,0]=ac0/wsum; out[y,x,1]=ac1/wsum; out[y,x,2]=ac2/wsum

def make_synthetic_pt(W=320, H=240, spp=4):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy, rad = W//2, H//2, min(W,H)//4
    dx, dy = xx-cx, yy-cy
    mask = dx*dx+dy*dy < rad*rad
    albedo = np.zeros((H,W,3), np.float32); albedo[:] = [0.2,0.2,0.25]
    albedo[mask] = [0.8,0.3,0.2]
    normal = np.zeros((H,W,3), np.float32); normal[:] = [0,1,0]
    normal[mask,0] = dx[mask]/rad
    normal[mask,1] = -dy[mask]/rad
    normal[mask,2] = np.sqrt(np.maximum(0, 1-(dx[mask]/rad)**2-(dy[mask]/rad)**2))
    L = np.array([0.4,0.7,0.3])
    ndl = np.clip((normal*L).sum(-1), 0, 1)
    gt = albedo * (0.15 + 0.85*ndl[...,None])
    rng = np.random.default_rng(0)
    noisy = np.zeros_like(gt)
    for _ in range(spp):
        noisy += gt * rng.uniform(0.3, 1.7, gt.shape)
    noisy /= spp
    return noisy.astype(np.float32), gt.astype(np.float32), albedo, normal

def main():
    out = Path(__file__).parent/"output"; out.mkdir(exist_ok=True)
    print("="*60); print("049 | 引导降噪（4 spp）"); print("="*60)
    noisy, gt, albedo, normal = make_synthetic_pt()
    H,W = noisy.shape[:2]
    d_n=cuda.to_device(noisy); d_a=cuda.to_device(albedo); d_nr=cuda.to_device(normal)
    d_o=cuda.device_array((H,W,3), np.float32)
    t0=time.perf_counter()
    joint_bilateral[((W+15)//16,(H+15)//16),(16,16)](d_n,d_a,d_nr,d_o,W,H,3.0,0.15,0.3)
    cuda.synchronize()
    ms=(time.perf_counter()-t0)*1000
    den=d_o.copy_to_host()
    def mse(a,b): return float(np.mean((a-b)**2))
    print(f"联合双边: {ms:.2f} ms | MSE noisy={mse(noisy,gt):.5f} denoise={mse(den,gt):.5f}")
    Image.fromarray((np.clip(noisy,0,1)*255).astype(np.uint8)).save(out/"noisy_4spp.png")
    Image.fromarray((np.clip(den,0,1)*255).astype(np.uint8)).save(out/"denoised.png")
    Image.fromarray((np.clip(gt,0,1)*255).astype(np.uint8)).save(out/"gt.png")
    (out/"tensorrt_interface.md").write_text(
        "# TensorRT 部署接口\n\n1. 训练 U-Net(9ch→3ch)\n2. ONNX→trtexec FP16\n3. 目标 1080p < 5ms\n", encoding="utf-8")
    print("✓ 降噪教学实现完成。")

if __name__=="__main__":
    main()
