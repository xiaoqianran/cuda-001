#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""033 | Lucas-Kanade 光流 + 金字塔 coarse-to-fine + 颜色编码。"""
from pathlib import Path
import numpy as np
from numba import cuda
from PIL import Image
import math

@cuda.jit(device=True)
def ci(v, lo, hi):
    if v < lo: return lo
    if v > hi: return hi
    return v

@cuda.jit
def lk_flow_kernel(I1, I2, flow, W, H, win):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x < win or y < win or x >= W - win or y >= H - win:
        return
    A00 = A01 = A11 = 0.0
    b0 = b1 = 0.0
    for dy in range(-win, win + 1):
        for dx in range(-win, win + 1):
            ix = x + dx
            iy = y + dy
            Ix = 0.5 * (I1[iy, ci(ix + 1, 0, W - 1)] - I1[iy, ci(ix - 1, 0, W - 1)])
            Iy = 0.5 * (I1[ci(iy + 1, 0, H - 1), ix] - I1[ci(iy - 1, 0, H - 1), ix])
            It = I2[iy, ix] - I1[iy, ix]
            A00 += Ix * Ix
            A01 += Ix * Iy
            A11 += Iy * Iy
            b0 -= Ix * It
            b1 -= Iy * It
    det = A00 * A11 - A01 * A01
    if abs(det) < 1e-3:
        flow[y, x, 0] = 0.0
        flow[y, x, 1] = 0.0
        return
    flow[y, x, 0] = (A11 * b0 - A01 * b1) / det
    flow[y, x, 1] = (-A01 * b0 + A00 * b1) / det

def pyramid_lk(I1, I2, levels=3, win=2):
    pyr1 = [I1.astype(np.float32)]
    pyr2 = [I2.astype(np.float32)]
    for _ in range(levels - 1):
        pyr1.append(pyr1[-1][::2, ::2].copy())
        pyr2.append(pyr2[-1][::2, ::2].copy())
    h, w = pyr1[-1].shape
    flow = np.zeros((h, w, 2), np.float32)
    for lev in range(levels - 1, -1, -1):
        J1, J2 = pyr1[lev], pyr2[lev]
        h, w = J1.shape
        if lev < levels - 1:
            flow = np.repeat(np.repeat(flow, 2, 0), 2, 1)[:h, :w] * 2
            if flow.shape[0] != h or flow.shape[1] != w:
                flow = np.zeros((h, w, 2), np.float32)
        d_I1 = cuda.to_device(J1)
        d_I2 = cuda.to_device(J2)
        d_f = cuda.to_device(flow.astype(np.float32))
        lk_flow_kernel[((w + 15) // 16, (h + 15) // 16), (16, 16)](d_I1, d_I2, d_f, w, h, win)
        flow = d_f.copy_to_host()
    return flow

def flow_to_color(flow):
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    ang = np.arctan2(flow[..., 1], flow[..., 0])
    hsv_h = (ang + np.pi) / (2 * np.pi)
    rgb = np.zeros(flow.shape[:2] + (3,), np.uint8)
    s = np.clip(mag / 3.0, 0, 1)
    v = np.clip(0.3 + mag / 5.0, 0, 1)
    # 快速 HSV→RGB 向量化近似
    h6 = hsv_h * 6
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    c = v * s
    x = c * (1 - np.abs(f - 1))  # not exact but ok for viz
    # simpler: map angle to hue via matplotlib-free formula
    for yy in range(0, flow.shape[0], 1):
        for xx in range(0, flow.shape[1], 1):
            hh = hsv_h[yy, xx] * 6
            ss, vv = float(s[yy, xx]), float(v[yy, xx])
            cc = vv * ss
            m = vv - cc
            hi = int(hh) % 6
            ff = hh - int(hh)
            if hi == 0: r, g, b = cc, cc * ff, 0
            elif hi == 1: r, g, b = cc * (1 - ff), cc, 0
            elif hi == 2: r, g, b = 0, cc, cc * ff
            elif hi == 3: r, g, b = 0, cc * (1 - ff), cc
            elif hi == 4: r, g, b = cc * ff, 0, cc
            else: r, g, b = cc, 0, cc * (1 - ff)
            rgb[yy, xx] = ((r + m) * 255, (g + m) * 255, (b + m) * 255)
    return rgb

def main():
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    print("=" * 60)
    print("033 | Lucas-Kanade 金字塔光流")
    print("=" * 60)
    W, H = 320, 240
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    I1 = (128 + 80 * np.sin(xx * 0.05) * np.cos(yy * 0.05)).astype(np.float32)
    shift = 2
    I2 = np.zeros_like(I1)
    I2[:, shift:] = I1[:, :-shift]
    I2[:, :shift] = I1[:, :shift]
    flow = pyramid_lk(I1, I2, levels=3, win=2)
    print(f"平均光流 ux={flow[...,0].mean():.3f} (期望≈{shift})")
    Image.fromarray(flow_to_color(flow)).save(out / "flow.png")
    Image.fromarray(I1.astype(np.uint8)).save(out / "frame1.png")
    print("✓ 光流估计完成。")

if __name__ == "__main__":
    main()
