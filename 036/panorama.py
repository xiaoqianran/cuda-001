#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""036 | 鱼眼→ERP 全景拼接 + 多频段融合 + HDR 多曝光。"""
from pathlib import Path
import numpy as np
from numba import cuda
from PIL import Image
import math

@cuda.jit
def fisheye_project(src, dst, mask, W, H, yaw_offset, fov):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= W or y >= H: return
    pi = 3.141592653589793
    lon = (x / float(W)) * 2 * pi - pi + yaw_offset
    lat = pi / 2 - (y / float(H)) * pi
    cl = math.cos(lat)
    dx = cl * math.sin(lon)
    dy = math.sin(lat)
    dz = cl * math.cos(lon)
    # acos clamp
    zz = dz
    if zz < -1.0: zz = -1.0
    if zz > 1.0: zz = 1.0
    theta = math.acos(zz)
    if theta > fov * 0.5:
        mask[y, x] = 0
        return
    phi = math.atan2(dy, dx)
    r = theta / (fov * 0.5)
    sh = src.shape[0]; sw = src.shape[1]
    u = (r * math.cos(phi) * 0.5 + 0.5) * (sw - 1)
    v = (r * math.sin(phi) * 0.5 + 0.5) * (sh - 1)
    ix = int(u); iy = int(v)
    if ix >= 0 and iy >= 0 and ix < sw and iy < sh:
        dst[y, x, 0] = src[iy, ix, 0]
        dst[y, x, 1] = src[iy, ix, 1]
        dst[y, x, 2] = src[iy, ix, 2]
        mask[y, x] = 1
    else:
        mask[y, x] = 0

def make_fisheye(size=256, color_shift=0):
    img = np.zeros((size, size, 3), np.uint8)
    cy = cx = size // 2
    for y in range(size):
        for x in range(size):
            dx, dy = (x-cx)/(size/2), (y-cy)/(size/2)
            r = math.sqrt(dx*dx+dy*dy)
            if r <= 1:
                img[y,x] = [int(128+100*dx+color_shift)%256, int(128+100*dy)%256, int(180-60*r)%256]
    return img

def multiband_blend(imgs, masks):
    h, w = imgs[0].shape[:2]
    acc = np.zeros((h, w, 3), np.float32)
    wacc = np.zeros((h, w), np.float32)
    for im, m in zip(imgs, masks):
        ww = m.astype(np.float32)
        for _ in range(12):
            pad = np.pad(ww, 1, mode='edge')
            nmax = np.maximum.reduce([pad[0:-2,1:-1], pad[2:,1:-1], pad[1:-1,0:-2], pad[1:-1,2:]])
            ww = np.maximum(ww, nmax * 0.9)
        acc += im.astype(np.float32) * ww[..., None]
        wacc += ww
    return np.clip(acc / (wacc[..., None] + 1e-6), 0, 255).astype(np.uint8)

def hdr_merge(exposures):
    acc = np.zeros_like(exposures[0], np.float32)
    wsum = np.zeros(exposures[0].shape[:2], np.float32)
    for e in exposures:
        f = e.astype(np.float32)
        gray = 0.299*f[:,:,0]+0.587*f[:,:,1]+0.114*f[:,:,2]
        w = np.exp(-((gray-128)**2)/(2*50**2))
        acc += f * w[..., None]
        wsum += w
    return np.clip(acc/(wsum[...,None]+1e-6), 0, 255).astype(np.uint8)

def main():
    out = Path(__file__).parent/"output"; out.mkdir(exist_ok=True)
    print("="*60); print("036 | 鱼眼拼接 ERP 全景"); print("="*60)
    W, H = 1024, 512
    fov = math.radians(180)
    imgs, masks = [], []
    for i, yaw in enumerate([-2.0, 0.0, 2.0]):
        fe = make_fisheye(256, color_shift=i*40)
        d_src = cuda.to_device(fe)
        d_dst = cuda.to_device(np.zeros((H,W,3), np.uint8))
        d_mask = cuda.to_device(np.zeros((H,W), np.uint8))
        fisheye_project[((W+15)//16,(H+15)//16),(16,16)](d_src,d_dst,d_mask,W,H,np.float32(yaw),np.float32(fov))
        imgs.append(d_dst.copy_to_host()); masks.append(d_mask.copy_to_host())
        Image.fromarray(imgs[-1]).save(out/f"cam{i}_erp.png")
    pano = multiband_blend(imgs, masks)
    Image.fromarray(pano).save(out/"panorama.png")
    exps = [np.clip(pano.astype(np.float32)*s,0,255).astype(np.uint8) for s in (0.5,1.0,2.0)]
    Image.fromarray(hdr_merge(exps)).save(out/"pano_hdr_merge.png")
    print(f"ERP: {W}x{H}")
    print("✓ 全景拼接完成。")

if __name__ == "__main__":
    main()
