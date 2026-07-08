#!/usr/bin/env python3
"""
001 | Lv.1 | Python + Numba CUDA
对一张 1080p 图像实现 GPU 版亮度调节和灰度化

核心收获：
  1. 用 @cuda.jit 写 kernel
  2. 把 grid/block/thread 的三维索引映射到二维像素 (x, y)
  3. 对比 NumPy CPU 与 Numba CUDA 的耗时，跑通 CPU → GPU 闭环
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from numba import cuda
from PIL import Image


# ---------------------------------------------------------------------------
# 工具：计时 / 校验
# ---------------------------------------------------------------------------

def timed(fn, *args, warmup: int = 1, repeats: int = 5, **kwargs):
    """简单计时：先 warmup，再取多次平均（秒）。"""
    for _ in range(warmup):
        fn(*args, **kwargs)
    if cuda.is_available():
        cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        out = fn(*args, **kwargs)
        if cuda.is_available():
            cuda.synchronize()  # 等 GPU 真正算完再停表
    elapsed = (time.perf_counter() - t0) / repeats
    return out, elapsed


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))


# ---------------------------------------------------------------------------
# CPU 参考实现（NumPy）
# ---------------------------------------------------------------------------

def cpu_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    """亮度调节：pixel * factor，再 clip 到 [0, 255]。"""
    out = img.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def cpu_grayscale(img: np.ndarray) -> np.ndarray:
    """标准亮度加权灰度：0.299 R + 0.587 G + 0.114 B。"""
    r = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    b = img[:, :, 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gray, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# GPU kernel
#
#  二维图像 (width, height) 映射到 CUDA 的 grid / block / thread：
#
#    blockDim  = (tx, ty, 1)     # 一个 block 里的线程排布
#    gridDim   = (gx, gy, 1)     # 需要多少个 block 盖满整图
#    threadIdx = (i, j, 0)       # block 内局部坐标
#    blockIdx  = (bx, by, 0)     # block 在 grid 中的坐标
#
#  全局像素坐标：
#    x = blockIdx.x * blockDim.x + threadIdx.x   # 列（宽方向）
#    y = blockIdx.y * blockDim.y + threadIdx.y   # 行（高方向）
#
#  边界检查必不可少：grid 往往略大于图像尺寸，
#  超出 (width, height) 的线程必须直接 return。
# ---------------------------------------------------------------------------

@cuda.jit
def brightness_kernel(src, dst, factor, width, height):
    """每个线程处理一个像素的 RGB 三通道。"""
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if x >= width or y >= height:
        return

    for c in range(3):  # R, G, B
        v = src[y, x, c] * factor
        # 手动 clamp（Numba CUDA 里没有 np.clip）
        if v < 0.0:
            v = 0.0
        elif v > 255.0:
            v = 255.0
        dst[y, x, c] = np.uint8(v)


@cuda.jit
def grayscale_kernel(src, dst, width, height):
    """每个线程把一个 RGB 像素变成一个灰度值。"""
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if x >= width or y >= height:
        return

    r = src[y, x, 0]
    g = src[y, x, 1]
    b = src[y, x, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    if gray < 0.0:
        gray = 0.0
    elif gray > 255.0:
        gray = 255.0
    dst[y, x] = np.uint8(gray)


def make_grid(width: int, height: int, block=(16, 16)):
    """
    根据图像尺寸和 block 大小计算 grid。

    常见写法：向上取整，保证每个像素至少被一个线程覆盖。
      grid_x = ceil(width  / block_x)
      grid_y = ceil(height / block_y)
    """
    bx, by = block
    gx = (width + bx - 1) // bx
    gy = (height + by - 1) // by
    return (gx, gy), (bx, by)


def gpu_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    height, width = img.shape[:2]
    d_src = cuda.to_device(img)
    d_dst = cuda.device_array_like(img)
    grid, block = make_grid(width, height)
    brightness_kernel[grid, block](d_src, d_dst, np.float32(factor), width, height)
    return d_dst.copy_to_host()


def gpu_grayscale(img: np.ndarray) -> np.ndarray:
    height, width = img.shape[:2]
    d_src = cuda.to_device(img)
    d_dst = cuda.device_array((height, width), dtype=np.uint8)
    grid, block = make_grid(width, height)
    grayscale_kernel[grid, block](d_src, d_dst, width, height)
    return d_dst.copy_to_host()


# ---------------------------------------------------------------------------
# 生成 / 加载测试图
# ---------------------------------------------------------------------------

def make_demo_image(width: int = 1920, height: int = 1080) -> np.ndarray:
    """
    合成一张 1080p 渐变图（无需外部文件）：
      - 水平：红色从 0→255
      - 垂直：绿色从 0→255
      - 蓝色：固定 128
    """
    ys = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    xs = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    r = np.broadcast_to(xs, (height, width))
    g = np.broadcast_to(ys, (height, width))
    b = np.full((height, width), 128.0, dtype=np.float32)
    img = np.stack([r, g, b], axis=-1)
    return np.clip(img, 0, 255).astype(np.uint8)


def load_or_make_image(path: Path | None, size=(1920, 1080)) -> np.ndarray:
    if path is not None and path.exists():
        img = Image.open(path).convert("RGB")
        img = img.resize(size, Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)
    return make_demo_image(*size)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("001 | GPU 亮度调节 & 灰度化  (Numba CUDA)")
    print("=" * 60)

    if not cuda.is_available():
        raise RuntimeError("未检测到可用的 CUDA 设备，请检查驱动 / GPU。")

    device = cuda.get_current_device()
    print(f"GPU      : {device.name.decode() if isinstance(device.name, bytes) else device.name}")
    print(f"CC       : {device.compute_capability}")
    print()

    # ---- 1. 准备 1080p 图像 ----
    img = load_or_make_image(None, size=(1920, 1080))
    h, w = img.shape[:2]
    print(f"图像尺寸 : {w} x {h}  (1080p), dtype={img.dtype}, shape={img.shape}")
    print(f"像素总数 : {w * h:,}  |  数据量 ≈ {img.nbytes / 1e6:.2f} MB")

    # 索引映射示意（一次说清楚）
    block = (16, 16)
    grid, block = make_grid(w, h, block)
    print()
    print("---- grid / block / thread → 像素映射 ----")
    print(f"  block = {block}           # 每个 block 16x16 = 256 线程")
    print(f"  grid  = {grid}            # 覆盖 1920x1080 需要的 block 数")
    print(f"  总线程 ≈ {grid[0] * block[0]} x {grid[1] * block[1]} "
          f"= {grid[0] * block[0] * grid[1] * block[1]:,}")
    print("  像素 (x, y) = (blockIdx.x * 16 + threadIdx.x,")
    print("                 blockIdx.y * 16 + threadIdx.y)")
    print("  若 x>=width 或 y>=height → 该线程直接 return（边界保护）")
    print()

    factor = 1.5  # 提亮 1.5 倍

    # ---- 2. 亮度调节：CPU vs GPU ----
    print("---- 亮度调节 (factor = 1.5) ----")
    cpu_bright, t_cpu_b = timed(cpu_brightness, img, factor, warmup=1, repeats=10)
    gpu_bright, t_gpu_b = timed(gpu_brightness, img, factor, warmup=2, repeats=20)
    diff_b = max_abs_diff(cpu_bright, gpu_bright)
    print(f"  CPU  (NumPy)  : {t_cpu_b * 1e3:8.3f} ms")
    print(f"  GPU  (Numba)  : {t_gpu_b * 1e3:8.3f} ms")
    print(f"  加速比        : {t_cpu_b / t_gpu_b:8.2f} x")
    print(f"  最大像素差    : {diff_b:.1f}  (应为 0 或极小)")
    print()

    # ---- 3. 灰度化：CPU vs GPU ----
    print("---- 灰度化 (0.299R + 0.587G + 0.114B) ----")
    cpu_gray, t_cpu_g = timed(cpu_grayscale, img, warmup=1, repeats=10)
    gpu_gray, t_gpu_g = timed(gpu_grayscale, img, warmup=2, repeats=20)
    diff_g = max_abs_diff(cpu_gray, gpu_gray)
    print(f"  CPU  (NumPy)  : {t_cpu_g * 1e3:8.3f} ms")
    print(f"  GPU  (Numba)  : {t_gpu_g * 1e3:8.3f} ms")
    print(f"  加速比        : {t_cpu_g / t_gpu_g:8.2f} x")
    print(f"  最大像素差    : {diff_g:.1f}  (浮点舍入允许 ≤1)")
    print()

    # ---- 4. 保存结果，方便肉眼检查 ----
    Image.fromarray(img).save(out_dir / "00_original.png")
    Image.fromarray(gpu_bright).save(out_dir / "01_brightness_gpu.png")
    Image.fromarray(gpu_gray).save(out_dir / "02_grayscale_gpu.png")
    print(f"结果已保存到: {out_dir}")
    print("  00_original.png")
    print("  01_brightness_gpu.png")
    print("  02_grayscale_gpu.png")
    print()
    print("✓ 第一个 CPU → GPU 闭环跑通。")
    print("=" * 60)


if __name__ == "__main__":
    main()
