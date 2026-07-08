#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""005 | GPU 直方图均衡化 + CLAHE（atomicAdd + 分块双线性合并）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from numba import cuda, int32, uint8, float32

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.image_utils import make_xray_like, timed, save_png

@cuda.jit
def hist_atomic_kernel(img, hist, n):
    """用 atomicAdd 统计 256-bin 直方图。"""
    i = cuda.grid(1)
    if i < n:
        v = int32(img[i])
        cuda.atomic.add(hist, v, 1)

@cuda.jit
def apply_lut_kernel(src, dst, lut, n):
    i = cuda.grid(1)
    if i < n:
        dst[i] = lut[src[i]]

def gpu_global_he(img2d: np.ndarray) -> np.ndarray:
    """全局直方图均衡化。"""
    flat = np.ascontiguousarray(img2d.ravel())
    n = flat.size
    d_img = cuda.to_device(flat)
    d_hist = cuda.to_device(np.zeros(256, dtype=np.int32))
    threads = 256
    blocks = (n + threads - 1) // threads
    hist_atomic_kernel[blocks, threads](d_img, d_hist, n)
    hist = d_hist.copy_to_host().astype(np.float64)
    # 前缀和 → CDF → LUT
    cdf = np.cumsum(hist)
    cdf_min = cdf[cdf > 0][0]
    lut = np.floor((cdf - cdf_min) / (n - cdf_min) * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    d_lut = cuda.to_device(lut)
    d_out = cuda.device_array_like(flat)
    apply_lut_kernel[blocks, threads](d_img, d_out, d_lut, n)
    return d_out.copy_to_host().reshape(img2d.shape)

def clahe_cpu_tiles(img, tile=8, clip_limit=2.0):
    """
    CLAHE 核心（教学实现）：
    分 tile 做限制对比度 HE，再用双线性插值合并映射。
    """
    h, w = img.shape
    th, tw = h // tile, w // tile
    # 每个 tile 一个 LUT
    luts = np.zeros((tile, tile, 256), dtype=np.uint8)
    for ty in range(tile):
        for tx in range(tile):
            y0, y1 = ty * th, (ty + 1) * th if ty < tile - 1 else h
            x0, x1 = tx * tw, (tx + 1) * tw if tx < tile - 1 else w
            patch = img[y0:y1, x0:x1]
            hist, _ = np.histogram(patch, bins=256, range=(0, 256))
            # clip
            cl = int(clip_limit * patch.size / 256.0)
            excess = 0
            for i in range(256):
                if hist[i] > cl:
                    excess += hist[i] - cl
                    hist[i] = cl
            hist += excess // 256
            cdf = np.cumsum(hist).astype(np.float64)
            cdf_min = cdf[cdf > 0][0] if np.any(cdf > 0) else 0
            denom = max(cdf[-1] - cdf_min, 1)
            luts[ty, tx] = np.floor((cdf - cdf_min) / denom * 255 + 0.5).clip(0, 255).astype(np.uint8)

    out = np.zeros_like(img)
    # 双线性插值四个相邻 tile 的映射
    for y in range(h):
        # tile 中心坐标
        gy = (y + 0.5) / th - 0.5
        y0 = int(np.floor(gy))
        y1 = y0 + 1
        fy = gy - y0
        y0 = max(0, min(tile - 1, y0))
        y1 = max(0, min(tile - 1, y1))
        for x in range(w):
            gx = (x + 0.5) / tw - 0.5
            x0 = int(np.floor(gx))
            x1 = x0 + 1
            fx = gx - x0
            x0 = max(0, min(tile - 1, x0))
            x1 = max(0, min(tile - 1, x1))
            v = int(img[y, x])
            v00 = float(luts[y0, x0, v])
            v01 = float(luts[y0, x1, v])
            v10 = float(luts[y1, x0, v])
            v11 = float(luts[y1, x1, v])
            out[y, x] = uint8_clip(
                (1 - fy) * ((1 - fx) * v00 + fx * v01) + fy * ((1 - fx) * v10 + fx * v11)
            )
    return out

def uint8_clip(v):
    return np.uint8(max(0, min(255, int(v + 0.5))))

@cuda.jit
def clahe_apply_kernel(img, luts, out, h, w, tile, th, tw):
    """GPU：双线性插值合并各 tile LUT。"""
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= w or y >= h:
        return
    gy = (y + 0.5) / th - 0.5
    y0 = int(gy)
    if gy < 0:
        y0 = int(gy) - 1
    y1 = y0 + 1
    fy = gy - y0
    if y0 < 0: y0 = 0
    if y1 < 0: y1 = 0
    if y0 >= tile: y0 = tile - 1
    if y1 >= tile: y1 = tile - 1
    gx = (x + 0.5) / tw - 0.5
    x0 = int(gx)
    if gx < 0:
        x0 = int(gx) - 1
    x1 = x0 + 1
    fx = gx - x0
    if x0 < 0: x0 = 0
    if x1 < 0: x1 = 0
    if x0 >= tile: x0 = tile - 1
    if x1 >= tile: x1 = tile - 1
    v = int32(img[y, x])
    v00 = float32(luts[y0, x0, v])
    v01 = float32(luts[y0, x1, v])
    v10 = float32(luts[y1, x0, v])
    v11 = float32(luts[y1, x1, v])
    val = (1 - fy) * ((1 - fx) * v00 + fx * v01) + fy * ((1 - fx) * v10 + fx * v11)
    if val < 0: val = 0
    if val > 255: val = 255
    out[y, x] = uint8(val)

def build_clahe_luts(img, tile=8, clip_limit=2.0):
    h, w = img.shape
    th, tw = max(1, h // tile), max(1, w // tile)
    luts = np.zeros((tile, tile, 256), dtype=np.uint8)
    for ty in range(tile):
        for tx in range(tile):
            y0, y1 = ty * th, (ty + 1) * th if ty < tile - 1 else h
            x0, x1 = tx * tw, (tx + 1) * tw if tx < tile - 1 else w
            patch = img[y0:y1, x0:x1]
            hist, _ = np.histogram(patch.ravel(), bins=256, range=(0, 256))
            cl = max(1, int(clip_limit * patch.size / 256.0))
            excess = int(np.maximum(hist - cl, 0).sum())
            hist = np.minimum(hist, cl)
            hist = hist + excess // 256
            cdf = np.cumsum(hist).astype(np.float64)
            nz = cdf[cdf > 0]
            cdf_min = nz[0] if len(nz) else 0.0
            denom = max(cdf[-1] - cdf_min, 1.0)
            luts[ty, tx] = np.floor((cdf - cdf_min) / denom * 255 + 0.5).clip(0, 255).astype(np.uint8)
    return luts, th, tw

def gpu_clahe(img, tile=8, clip_limit=2.0):
    h, w = img.shape
    luts, th, tw = build_clahe_luts(img, tile, clip_limit)
    d_img = cuda.to_device(np.ascontiguousarray(img))
    d_luts = cuda.to_device(luts)
    d_out = cuda.device_array_like(img)
    grid = ((w + 15) // 16, (h + 15) // 16)
    clahe_apply_kernel[grid, (16, 16)](d_img, d_luts, d_out, h, w, tile, th, tw)
    return d_out.copy_to_host()

def main():
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    print("=" * 60)
    print("005 | 直方图均衡化 + CLAHE (atomicAdd)")
    print("=" * 60)
    xray = make_xray_like(512, 512)
    he, t_he = timed(gpu_global_he, xray, warmup=1, repeats=20)
    clahe, t_c = timed(gpu_clahe, xray, 8, 2.0, warmup=1, repeats=10)
    print(f"全局 HE: {t_he*1e3:.3f} ms | CLAHE: {t_c*1e3:.3f} ms")
    print(f"原图对比度(std)={xray.std():.2f} HE={he.std():.2f} CLAHE={clahe.std():.2f}")
    save_png(out / "xray.png", xray)
    save_png(out / "he.png", he)
    save_png(out / "clahe.png", clahe)
    print("✓ atomic 直方图 + CLAHE 双线性合并完成。")

if __name__ == "__main__":
    main()
