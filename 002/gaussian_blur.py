#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
002 | Lv.1 | GPU 高斯模糊
共享内存 tile + halo 卷积；clamp/reflect/wrap 边界；5x5 与 15x15 对比。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from numba import cuda, float32, uint8, int32
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.image_utils import make_demo_rgb, timed, save_png, grid_for

# 边界模式：0=clamp, 1=reflect, 2=wrap
MODE_CLAMP, MODE_REFLECT, MODE_WRAP = 0, 1, 2

def make_gaussian_kernel(ksize: int, sigma: float | None = None) -> np.ndarray:
    """生成归一化二维高斯核。"""
    if sigma is None:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    r = ksize // 2
    ax = np.arange(-r, r + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    ker = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    ker /= ker.sum()
    return ker.astype(np.float32)

def cpu_blur(img: np.ndarray, kernel: np.ndarray, mode: int = MODE_CLAMP) -> np.ndarray:
    """CPU 参考：直接卷积。"""
    h, w = img.shape[:2]
    r = kernel.shape[0] // 2
    out = np.zeros_like(img, dtype=np.float32)
    src = img.astype(np.float32)

    def idx(p, n):
        if mode == MODE_CLAMP:
            return max(0, min(n - 1, p))
        if mode == MODE_WRAP:
            return p % n
        # reflect
        if p < 0:
            return -p
        if p >= n:
            return 2 * n - p - 2
        return p

    for y in range(h):
        for x in range(w):
            acc = np.zeros(3, dtype=np.float32)
            for ky in range(-r, r + 1):
                for kx in range(-r, r + 1):
                    iy, ix = idx(y + ky, h), idx(x + kx, w)
                    acc += src[iy, ix] * kernel[ky + r, kx + r]
            out[y, x] = acc
    return np.clip(out, 0, 255).astype(np.uint8)

# 最大支持核半径（15x15 → r=7）
MAX_R = 7
# block 16x16，共享内存 tile = 16+2*MAX_R
TILE = 16
SH_W = TILE + 2 * MAX_R
SH_H = TILE + 2 * MAX_R

@cuda.jit(device=True)
def border_index(p, n, mode):
    """设备端边界索引映射。"""
    if mode == 0:  # clamp
        if p < 0:
            return 0
        if p >= n:
            return n - 1
        return p
    if mode == 2:  # wrap
        return p % n
    # reflect
    if p < 0:
        return -p
    if p >= n:
        return 2 * n - p - 2
    return p

@cuda.jit
def gaussian_blur_shared_kernel(src, dst, kernel, ksize, width, height, mode):
    """
    共享内存高斯卷积：
    1) 每个 block 协作加载 (TILE+2r)×(TILE+2r) 的 halo tile 到 shared
    2) 同步后每个线程对中心 TILE 区域做卷积
    """
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    # 全局输出像素
    x = cuda.blockIdx.x * TILE + tx
    y = cuda.blockIdx.y * TILE + ty
    r = ksize // 2

    # 动态共享内存：三通道 float，按 [c][sy][sx] 展开为一维
    sm = cuda.shared.array(shape=(3, SH_H, SH_W), dtype=float32)

    # block 原点（无 halo）在全局图中的位置
    base_x = cuda.blockIdx.x * TILE - r
    base_y = cuda.blockIdx.y * TILE - r

    # 协作加载：用 16x16 线程覆盖更大的 SH 区域（循环铺满）
    for sy in range(ty, SH_H, TILE):
        for sx in range(tx, SH_W, TILE):
            gx = border_index(base_x + sx, width, mode)
            gy = border_index(base_y + sy, height, mode)
            for c in range(3):
                sm[c, sy, sx] = src[gy, gx, c]

    cuda.syncthreads()

    if x >= width or y >= height:
        return

    # 卷积：线程中心在 shared 中的位置是 (tx+r, ty+r)
    for c in range(3):
        acc = float32(0.0)
        for ky in range(ksize):
            for kx in range(ksize):
                acc += sm[c, ty + ky, tx + kx] * kernel[ky, kx]
        if acc < 0.0:
            acc = 0.0
        elif acc > 255.0:
            acc = 255.0
        dst[y, x, c] = uint8(acc)

def gpu_blur(img: np.ndarray, kernel: np.ndarray, mode: int = MODE_CLAMP) -> np.ndarray:
    """Host 封装：上传、launch、回读。"""
    h, w = img.shape[:2]
    ksize = kernel.shape[0]
    d_src = cuda.to_device(np.ascontiguousarray(img))
    d_dst = cuda.device_array_like(img)
    d_ker = cuda.to_device(np.ascontiguousarray(kernel))
    # grid 按 TILE 划分
    grid = ((w + TILE - 1) // TILE, (h + TILE - 1) // TILE)
    block = (TILE, TILE)
    gaussian_blur_shared_kernel[grid, block](d_src, d_dst, d_ker, ksize, w, h, mode)
    return d_dst.copy_to_host()

def main():
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(exist_ok=True)
    print("=" * 60)
    print("002 | GPU 高斯模糊 (shared memory tile + halo)")
    print("=" * 60)
    assert cuda.is_available()
    img = make_demo_rgb(1920, 1080)
    print(f"图像: {img.shape}, GPU: {cuda.get_current_device().name}")

    modes = [("clamp", MODE_CLAMP), ("reflect", MODE_REFLECT), ("wrap", MODE_WRAP)]
    for name, ksize in [("5x5", 5), ("15x15", 15)]:
        ker = make_gaussian_kernel(ksize)
        print(f"\n---- 核 {name} (sigma≈自动) ----")
        # 小图 CPU 对照（全 1080p CPU 双重循环太慢，用 256 切片校验）
        patch = img[400:656, 800:1056].copy()
        cpu_out = cpu_blur(patch, ker, MODE_CLAMP)
        gpu_patch = gpu_blur(patch, ker, MODE_CLAMP)
        diff = float(np.max(np.abs(cpu_out.astype(np.int16) - gpu_patch.astype(np.int16))))
        print(f"  256 切片 CPU/GPU 最大差: {diff}")

        for mname, m in modes:
            _, t = timed(gpu_blur, img, ker, m, warmup=1, repeats=10)
            print(f"  GPU mode={mname:7s}: {t*1e3:8.3f} ms")

        result = gpu_blur(img, ker, MODE_CLAMP)
        save_png(out_dir / f"blur_{name}.png", result)
        print(f"  已保存 blur_{name}.png")

    save_png(out_dir / "original.png", img)
    print("\n✓ tile + halo 共享内存卷积跑通。")

if __name__ == "__main__":
    main()
