# -*- coding: utf-8 -*-
"""公共图像工具：程序化测试图、计时、保存。"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

try:
    from numba import cuda

    _HAS_CUDA = cuda.is_available()
except Exception:
    _HAS_CUDA = False


def make_demo_rgb(width: int = 1920, height: int = 1080) -> np.ndarray:
    """合成 RGB 渐变测试图（uint8, HWC）。"""
    ys = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    xs = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    r = np.broadcast_to(xs, (height, width))
    g = np.broadcast_to(ys, (height, width))
    b = np.full((height, width), 128.0, dtype=np.float32)
    # 叠加棋盘与圆形，方便看滤波/边缘效果
    yy, xx = np.mgrid[0:height, 0:width]
    checker = (((xx // 64) + (yy // 64)) % 2) * 40.0
    cx, cy = width // 2, height // 2
    disk = ((xx - cx) ** 2 + (yy - cy) ** 2) < (min(width, height) // 5) ** 2
    img = np.stack([r + checker, g + checker, b], axis=-1)
    img[disk] = [220, 40, 40]
    return np.clip(img, 0, 255).astype(np.uint8)


def make_demo_gray(width: int = 1920, height: int = 1080) -> np.ndarray:
    """灰度测试图。"""
    rgb = make_demo_rgb(width, height).astype(np.float32)
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)


def make_xray_like(width: int = 512, height: int = 512) -> np.ndarray:
    """模拟医学 X 光低对比度灰度图（CLAHE 测试用）。"""
    yy, xx = np.ogrid[0:height, 0:width]
    base = 80 + 30 * np.sin(xx / 40.0) * np.cos(yy / 50.0)
    bone = 40 * np.exp(-((xx - width * 0.5) ** 2 + (yy - height * 0.45) ** 2) / (2 * 80**2))
    noise = np.random.default_rng(0).normal(0, 5, (height, width))
    img = np.clip(base + bone + noise, 0, 255).astype(np.uint8)
    return img


def make_hdr_float(width: int = 960, height: int = 540) -> np.ndarray:
    """合成 HDR 浮点 RGB（可超 1.0），用于 tone mapping。"""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = width / 2, height / 2
    # 亮光源
    light = 20.0 * np.exp(-((xx - cx) ** 2 + (yy - cy * 0.4) ** 2) / (2 * 40**2))
    floor = 0.1 + 0.3 * (yy / height)
    r = floor + light
    g = floor * 0.9 + light * 0.85
    b = floor * 0.7 + light * 0.6
    # 高光反射带
    band = 5.0 * np.exp(-((yy - height * 0.7) ** 2) / (2 * 8**2))
    return np.stack([r + band, g + band * 0.9, b + band * 0.5], axis=-1).astype(np.float32)


def timed(fn, *args, warmup: int = 1, repeats: int = 5, sync_cuda: bool = True, **kwargs):
    """多次计时取平均（秒）。GPU 路径末尾可同步。"""
    for _ in range(warmup):
        fn(*args, **kwargs)
    if sync_cuda and _HAS_CUDA:
        cuda.synchronize()
    t0 = time.perf_counter()
    out = None
    for _ in range(repeats):
        out = fn(*args, **kwargs)
        if sync_cuda and _HAS_CUDA:
            cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / repeats
    return out, elapsed


def save_png(path: Path | str, arr: np.ndarray) -> None:
    """保存 uint8 图像为 PNG。"""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = arr
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        Image.fromarray(a, mode="L").save(path)
    else:
        Image.fromarray(a[..., :3], mode="RGB").save(path)


def grid_for(width: int, height: int, block=(16, 16)):
    """计算覆盖二维图像的 CUDA grid。"""
    bx, by = block
    return ((width + bx - 1) // bx, (height + by - 1) // by), (bx, by)
