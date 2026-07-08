#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""003 | GPU Sobel 边缘检测：单 kernel 完成 Gx/Gy + 幅值 + 可选二值化。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from numba import cuda, float32, uint8
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.image_utils import make_demo_rgb, timed, save_png

@cuda.jit
def sobel_kernel(gray, mag, binary, width, height, threshold, do_binary):
    """
    每个线程处理一个像素：
    - 3x3 Sobel Gx / Gy
    - 幅值 = sqrt(Gx^2 + Gy^2)
    - 可选阈值二值化
    """
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= width or y >= height:
        return
    # 边界 clamp 采样
    def sample(ix, iy):
        if ix < 0: ix = 0
        if iy < 0: iy = 0
        if ix >= width: ix = width - 1
        if iy >= height: iy = height - 1
        return float32(gray[iy, ix])

    # Sobel 核
    # Gx: [[-1,0,1],[-2,0,2],[-1,0,1]]
    # Gy: [[-1,-2,-1],[0,0,0],[1,2,1]]
    gx = (
        -sample(x-1, y-1) + sample(x+1, y-1)
        - 2 * sample(x-1, y) + 2 * sample(x+1, y)
        - sample(x-1, y+1) + sample(x+1, y+1)
    )
    gy = (
        -sample(x-1, y-1) - 2 * sample(x, y-1) - sample(x+1, y-1)
        + sample(x-1, y+1) + 2 * sample(x, y+1) + sample(x+1, y+1)
    )
    m = (gx * gx + gy * gy) ** 0.5
    if m > 255.0:
        m = 255.0
    mag[y, x] = uint8(m)
    if do_binary:
        binary[y, x] = uint8(255 if m >= threshold else 0)

def to_gray(img):
    f = img.astype(np.float32)
    return (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]).astype(np.uint8)

def gpu_sobel(gray, threshold=80.0, do_binary=True):
    h, w = gray.shape
    d_g = cuda.to_device(gray)
    d_m = cuda.device_array((h, w), dtype=np.uint8)
    d_b = cuda.device_array((h, w), dtype=np.uint8)
    block = (16, 16)
    grid = ((w + 15) // 16, (h + 15) // 16)
    sobel_kernel[grid, block](d_g, d_m, d_b, w, h, np.float32(threshold), 1 if do_binary else 0)
    return d_m.copy_to_host(), d_b.copy_to_host()

def cpu_sobel_opencv(gray, threshold=80):
    try:
        import cv2
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        mag = np.clip(mag, 0, 255).astype(np.uint8)
        _, binary = cv2.threshold(mag, threshold, 255, cv2.THRESH_BINARY)
        return mag, binary
    except Exception as e:
        print("OpenCV 不可用，跳过 CPU 对照:", e)
        return None, None

def main():
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    print("=" * 60)
    print("003 | GPU Sobel 边缘检测")
    print("=" * 60)
    img = make_demo_rgb(1920, 1080)
    gray = to_gray(img)
    thr = 80.0
    (mag, binary), t_gpu = timed(gpu_sobel, gray, thr, True, warmup=2, repeats=20)
    print(f"GPU Sobel+二值化: {t_gpu*1e3:.3f} ms  (threshold={thr})")
    cv_mag, cv_bin = cpu_sobel_opencv(gray, int(thr))
    if cv_mag is not None:
        _, t_cpu = timed(cpu_sobel_opencv, gray, int(thr), warmup=1, repeats=10, sync_cuda=False)
        diff = float(np.mean(np.abs(mag.astype(np.float32) - cv_mag.astype(np.float32))))
        print(f"OpenCV CPU: {t_cpu*1e3:.3f} ms | 平均幅值差: {diff:.3f}")
        print(f"加速比: {t_cpu/t_gpu:.2f}x")
    save_png(out / "gray.png", gray)
    save_png(out / "magnitude.png", mag)
    save_png(out / "binary.png", binary)
    print("已保存 magnitude/binary.png")
    print("✓ 单 kernel 融合 Gx/Gy/幅值/二值化 完成。")

if __name__ == "__main__":
    main()
