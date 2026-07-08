#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""006 | 多 Pass GPU 图像管线：显存驻留，避免 host-device 来回拷贝。"""
from __future__ import annotations
import sys, time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, List
import numpy as np
from numba import cuda, float32, uint8, int32

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.image_utils import make_demo_rgb, save_png


@cuda.jit(device=True)
def clamp_i(v, lo, hi):
    """整数边界钳制（避免 min/max 多态问题）。"""
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


@cuda.jit
def median3_kernel(src, dst, w, h):
    """3x3 中值滤波（去噪），RGB 分通道。"""
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= w or y >= h:
        return
    for c in range(3):
        # 收集 9 邻域（边界 clamp）
        vals = cuda.local.array(9, float32)
        k = 0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ix = clamp_i(x + dx, 0, w - 1)
                iy = clamp_i(y + dy, 0, h - 1)
                vals[k] = src[iy, ix, c]
                k += 1
        # 简单插入排序取中值
        for i in range(9):
            for j in range(i + 1, 9):
                if vals[j] < vals[i]:
                    t = vals[i]
                    vals[i] = vals[j]
                    vals[j] = t
        dst[y, x, c] = uint8(vals[4])


@cuda.jit
def color_correct_kernel(src, dst, w, h, contrast, brightness, sat):
    """色彩校正：对比度/亮度/饱和度。"""
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= w or y >= h:
        return
    r = (src[y, x, 0] / 255.0 - 0.5) * contrast + 0.5 + brightness
    g = (src[y, x, 1] / 255.0 - 0.5) * contrast + 0.5 + brightness
    b = (src[y, x, 2] / 255.0 - 0.5) * contrast + 0.5 + brightness
    # 饱和度：向灰度插值
    yv = 0.299 * r + 0.587 * g + 0.114 * b
    r = yv + (r - yv) * sat
    g = yv + (g - yv) * sat
    b = yv + (b - yv) * sat
    # 手动写三通道，避免 enumerate 元组
    if r < 0.0:
        r = 0.0
    if r > 1.0:
        r = 1.0
    if g < 0.0:
        g = 0.0
    if g > 1.0:
        g = 1.0
    if b < 0.0:
        b = 0.0
    if b > 1.0:
        b = 1.0
    dst[y, x, 0] = uint8(r * 255.0)
    dst[y, x, 1] = uint8(g * 255.0)
    dst[y, x, 2] = uint8(b * 255.0)


@cuda.jit
def sharpen_kernel(src, dst, w, h, amount):
    """非锐化掩模：out = src + amount*(src - blur3x3)。"""
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= w or y >= h:
        return
    for c in range(3):
        acc = float32(0.0)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ix = clamp_i(x + dx, 0, w - 1)
                iy = clamp_i(y + dy, 0, h - 1)
                acc += src[iy, ix, c]
        blur = acc / 9.0
        v = src[y, x, c] + amount * (src[y, x, c] - blur)
        if v < 0.0:
            v = 0.0
        if v > 255.0:
            v = 255.0
        dst[y, x, c] = uint8(v)


@cuda.jit
def reinhard_u8_kernel(src, dst, w, h, exposure):
    """对 LDR 伪 HDR（/255 * exposure）做简易 Reinhard。"""
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= w or y >= h:
        return
    for c in range(3):
        v = src[y, x, c] / 255.0 * exposure
        v = v / (1.0 + v)
        # gamma 2.2
        v = v ** (1.0 / 2.2)
        v = v * 255.0
        if v > 255.0:
            v = 255.0
        dst[y, x, c] = uint8(v)


@dataclass
class Pass:
    name: str
    fn: Callable  # (d_in, d_out, w, h, grid, block, params) -> None
    params: dict = field(default_factory=dict)


class GpuPipeline:
    """多 pass 管线：中间结果全程驻留显存。"""
    def __init__(self, passes: List[Pass]):
        self.passes = passes
        self.timings = {}

    def run(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        d_a = cuda.to_device(np.ascontiguousarray(img))
        d_b = cuda.device_array_like(img)
        src, dst = d_a, d_b
        block = (16, 16)
        grid = ((w + 15) // 16, (h + 15) // 16)
        for p in self.passes:
            cuda.synchronize()
            t0 = time.perf_counter()
            p.fn(src, dst, w, h, grid, block, p.params)
            cuda.synchronize()
            self.timings[p.name] = time.perf_counter() - t0
            src, dst = dst, src  # 乒乓缓冲，零 host 拷贝
        return src.copy_to_host()


def make_default_pipeline(order=None) -> GpuPipeline:
    def median(src, dst, w, h, grid, block, params):
        median3_kernel[grid, block](src, dst, w, h)

    def color(src, dst, w, h, grid, block, params):
        color_correct_kernel[grid, block](
            src, dst, w, h,
            np.float32(params.get("contrast", 1.1)),
            np.float32(params.get("brightness", 0.02)),
            np.float32(params.get("sat", 1.15)),
        )

    def sharp(src, dst, w, h, grid, block, params):
        sharpen_kernel[grid, block](src, dst, w, h, np.float32(params.get("amount", 0.8)))

    def tone(src, dst, w, h, grid, block, params):
        reinhard_u8_kernel[grid, block](src, dst, w, h, np.float32(params.get("exposure", 1.2)))

    catalog = {
        "median": Pass("median", median),
        "color": Pass("color", color, {"contrast": 1.1, "brightness": 0.02, "sat": 1.15}),
        "sharpen": Pass("sharpen", sharp, {"amount": 0.8}),
        "tone": Pass("tone", tone, {"exposure": 1.2}),
    }
    if order is None:
        order = ["median", "color", "sharpen", "tone"]
    return GpuPipeline([catalog[k] for k in order])


def main():
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    print("=" * 60)
    print("006 | 多 Pass GPU 管线（显存驻留）")
    print("=" * 60)
    img = make_demo_rgb(1280, 720)
    for name, order in [
        ("full", ["median", "color", "sharpen", "tone"]),
        ("sharp_tone", ["sharpen", "tone"]),
    ]:
        pipe = make_default_pipeline(order)
        t0 = time.perf_counter()
        result = pipe.run(img)
        total = time.perf_counter() - t0
        print(f"\n管线 [{name}] order={order}")
        for k, v in pipe.timings.items():
            print(f"  pass {k:10s}: {v*1e3:7.3f} ms")
        print(f"  总计: {total*1e3:.3f} ms")
        save_png(out / f"pipeline_{name}.png", result)
    save_png(out / "input.png", img)
    print("\n✓ 多 pass 显存驻留管线跑通。")


if __name__ == "__main__":
    main()
