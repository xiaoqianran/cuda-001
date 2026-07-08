#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""004 | HDR 色调映射：Reinhard、ACES、局部自适应（双边网格简化）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from numba import cuda, float32, uint8

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.image_utils import make_hdr_float, timed, save_png

@cuda.jit(device=True)
def luminance(r, g, b):
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

@cuda.jit(device=True)
def clamp01(v):
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v

@cuda.jit
def reinhard_kernel(hdr, ldr, width, height, exposure, gamma, white):
    """全局 Reinhard: L' = L*(1+L/white^2)/(1+L)，再 gamma。"""
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= width or y >= height:
        return
    r = hdr[y, x, 0] * exposure
    g = hdr[y, x, 1] * exposure
    b = hdr[y, x, 2] * exposure
    L = luminance(r, g, b)
    if L < 1e-8:
        scale = 1.0
    else:
        Lp = L * (1.0 + L / (white * white)) / (1.0 + L)
        scale = Lp / L
    inv_g = 1.0 / gamma
    for c in range(3):
        if c == 0:
            v = r * scale
        elif c == 1:
            v = g * scale
        else:
            v = b * scale
        v = clamp01(v)
        v = v ** inv_g
        ldr[y, x, c] = uint8(v * 255.0 + 0.5)

@cuda.jit
def aces_kernel(hdr, ldr, width, height, exposure, gamma):
    """ACES filmic 近似（Narkowicz）。"""
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= width or y >= height:
        return
    a, bcoef, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    inv_g = 1.0 / gamma
    for ch in range(3):
        v = hdr[y, x, ch] * exposure
        v = (v * (a * v + bcoef)) / (v * (c * v + d) + e)
        v = clamp01(v)
        v = v ** inv_g
        ldr[y, x, ch] = uint8(v * 255.0 + 0.5)

@cuda.jit
def local_reinhard_kernel(hdr, log_lum_blur, ldr, width, height, exposure, gamma, white):
    """局部自适应：用预模糊 log 亮度近似局部均值。"""
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= width or y >= height:
        return
    r = hdr[y, x, 0] * exposure
    g = hdr[y, x, 1] * exposure
    b = hdr[y, x, 2] * exposure
    L = luminance(r, g, b) + 1e-6
    Lavg = float32(log_lum_blur[y, x])
    Lp = L / (1.0 + Lavg)
    Lp = Lp * (1.0 + Lp / (white * white))
    scale = Lp / L
    inv_g = 1.0 / gamma
    for c in range(3):
        if c == 0:
            v = r * scale
        elif c == 1:
            v = g * scale
        else:
            v = b * scale
        v = clamp01(v)
        v = v ** inv_g
        ldr[y, x, c] = uint8(v * 255.0 + 0.5)

def box_blur(img2d, k=15):
    """可分离盒式模糊（局部均值近似）。"""
    pad = k // 2
    # 水平
    p = np.pad(img2d, ((0, 0), (pad, pad)), mode="edge")
    cs = np.cumsum(p, axis=1)
    hblur = (cs[:, k:] - cs[:, :-k]) / k
    # 垂直
    p2 = np.pad(hblur, ((pad, pad), (0, 0)), mode="edge")
    cs2 = np.cumsum(p2, axis=0)
    return ((cs2[k:, :] - cs2[:-k, :]) / k).astype(np.float32)

def launch(kernel, hdr, *extra):
    h, w = hdr.shape[:2]
    d_hdr = cuda.to_device(np.ascontiguousarray(hdr))
    d_ldr = cuda.device_array((h, w, 3), dtype=np.uint8)
    grid = ((w + 15) // 16, (h + 15) // 16)
    block = (16, 16)
    kernel[grid, block](d_hdr, d_ldr, w, h, *extra)
    return d_ldr.copy_to_host()

def main():
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    print("=" * 60)
    print("004 | GPU Tone Mapping (Reinhard / ACES / Local)")
    print("=" * 60)
    hdr = make_hdr_float(960, 540)
    print(f"HDR shape={hdr.shape} range=[{hdr.min():.3f}, {hdr.max():.3f}]")
    exposure, gamma, white = 1.0, 2.2, 2.0

    reinhard, t1 = timed(launch, reinhard_kernel, hdr, np.float32(exposure), np.float32(gamma), np.float32(white), warmup=1, repeats=20)
    aces, t2 = timed(launch, aces_kernel, hdr, np.float32(exposure), np.float32(gamma), warmup=1, repeats=20)

    L = 0.2126 * hdr[..., 0] + 0.7152 * hdr[..., 1] + 0.0722 * hdr[..., 2]
    logL = np.log(L * exposure + 1e-6).astype(np.float32)
    blur = box_blur(logL, 21).astype(np.float32)

    def local_map():
        h, w = hdr.shape[:2]
        d_hdr = cuda.to_device(hdr)
        d_blur = cuda.to_device(blur)
        d_ldr = cuda.device_array((h, w, 3), dtype=np.uint8)
        grid = ((w + 15) // 16, (h + 15) // 16)
        local_reinhard_kernel[grid, (16, 16)](d_hdr, d_blur, d_ldr, w, h, np.float32(exposure), np.float32(gamma), np.float32(white))
        return d_ldr.copy_to_host()

    local, t3 = timed(local_map, warmup=1, repeats=10)
    print(f"Reinhard: {t1*1e3:.3f} ms | ACES: {t2*1e3:.3f} ms | Local: {t3*1e3:.3f} ms")
    hdr_vis = np.clip(np.log1p(hdr) / np.log1p(hdr.max()) * 255, 0, 255).astype(np.uint8)
    save_png(out / "hdr_vis.png", hdr_vis)
    save_png(out / "reinhard.png", reinhard)
    save_png(out / "aces.png", aces)
    save_png(out / "local_reinhard.png", local)
    print("已保存 reinhard/aces/local_reinhard.png")
    print("✓ HDR→LDR 色调映射闭环完成。")

if __name__ == "__main__":
    main()
