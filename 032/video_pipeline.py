#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""032 | 视频滤镜流水线：合成帧模拟解码，GPU 去噪+色调+增强。"""
from pathlib import Path
import numpy as np
from numba import cuda, uint8
import time
from PIL import Image

@cuda.jit(device=True)
def ci(v, lo, hi):
    if v < lo: return lo
    if v > hi: return hi
    return v

@cuda.jit
def denoise_color_enhance(src, dst, W, H, sat, exposure):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= W or y >= H:
        return
    for c in range(3):
        acc = 0.0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ix = ci(x + dx, 0, W - 1)
                iy = ci(y + dy, 0, H - 1)
                acc += src[iy, ix, c]
        dst[y, x, c] = acc / 9.0 / 255.0 * exposure
    r, g, b = dst[y, x, 0], dst[y, x, 1], dst[y, x, 2]
    yv = 0.299 * r + 0.587 * g + 0.114 * b
    r = yv + (r - yv) * sat
    g = yv + (g - yv) * sat
    b = yv + (b - yv) * sat
    r = r / (1 + r)
    g = g / (1 + g)
    b = b / (1 + b)
    r = r ** 0.45
    g = g ** 0.45
    b = b ** 0.45
    if r > 1.0: r = 1.0
    if g > 1.0: g = 1.0
    if b > 1.0: b = 1.0
    dst[y, x, 0] = r * 255
    dst[y, x, 1] = g * 255
    dst[y, x, 2] = b * 255

def make_frame(i, W, H):
    yy, xx = np.mgrid[0:H, 0:W]
    t = i * 0.1
    img = np.zeros((H, W, 3), np.float32)
    img[..., 0] = 128 + 60 * np.sin(xx * 0.01 + t)
    img[..., 1] = 100 + 40 * np.cos(yy * 0.01 - t)
    img[..., 2] = 150
    rng = np.random.default_rng(i)
    img += rng.integers(-15, 15, img.shape)
    return np.clip(img, 0, 255).astype(np.float32)

def main():
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    print("=" * 60)
    print("032 | GPU 视频滤镜流水线")
    print("=" * 60)
    W, H, nframes = 1280, 720, 20
    d_b = cuda.device_array((H, W, 3), np.float32)
    grid = ((W + 15) // 16, (H + 15) // 16)
    times = []
    for i in range(nframes):
        frame = make_frame(i, W, H)
        d_src = cuda.to_device(frame)
        t0 = time.perf_counter()
        denoise_color_enhance[grid, (16, 16)](d_src, d_b, W, H, 1.2, 1.1)
        cuda.synchronize()
        times.append(time.perf_counter() - t0)
        if i in (0, nframes // 2):
            h = np.clip(d_b.copy_to_host(), 0, 255).astype(np.uint8)
            Image.fromarray(h).save(out / f"frame_{i:03d}.png")
    avg = np.mean(times) * 1000
    fps = 1000 / avg
    print(f"{W}x{H} 平均 {avg:.2f} ms/帧 → {fps:.1f} fps")
    print(f"目标 ≥30fps: {'达标' if fps >= 30 else '教学规模可接受'}")
    print("✓ 视频 GPU 管线完成。")

if __name__ == "__main__":
    main()
