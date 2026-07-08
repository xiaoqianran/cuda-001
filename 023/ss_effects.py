#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""023 | 屏幕空间效果：SSAO / SSR 简化 / Bloom / ToneMap / FXAA，可开关计时。"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
from numba import cuda, float32, uint8
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.image_utils import save_png

def make_gbuffer(W=640, H=360):
    """程序化 G-Buffer：深度、法线、颜色。"""
    depth = np.full((H, W), 10.0, np.float32)
    normal = np.zeros((H, W, 3), np.float32)
    normal[..., 1] = 1.0
    color = np.zeros((H, W, 3), np.float32)
    color[:] = [0.15, 0.17, 0.22]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = xx / W * 2 - 1
    v = 1 - yy / H * 2
    for y in range(H):
        for x in range(W):
            rd = np.array([u[y, x] * W / H, v[y, x], -1.5], np.float32)
            rd /= np.linalg.norm(rd) + 1e-8
            eye = np.array([0, 0.8, 2.0], np.float32)
            if abs(rd[1]) > 1e-5:
                t = -eye[1] / rd[1]
                if 0.1 < t < depth[y, x]:
                    depth[y, x] = t
                    normal[y, x] = [0, 1, 0]
                    p = eye + rd * t
                    color[y, x] = [0.6, 0.6, 0.6] if (int(p[0]) + int(p[2])) % 2 == 0 else [0.35, 0.35, 0.35]
            c = np.array([0, 0.5, -1.0], np.float32)
            r = 0.5
            oc = eye - c
            a = np.dot(rd, rd)
            b = 2 * np.dot(oc, rd)
            cc = np.dot(oc, oc) - r * r
            disc = b * b - 4 * a * cc
            if disc > 0:
                t = (-b - np.sqrt(disc)) / (2 * a)
                if 0.1 < t < depth[y, x]:
                    depth[y, x] = t
                    p = eye + rd * t
                    normal[y, x] = (p - c) / r
                    color[y, x] = [0.8, 0.25, 0.2]
    return depth, normal, color

@cuda.jit(device=True)
def clamp_i(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v

@cuda.jit
def ssao_kernel(depth, normal, out, W, H, radius, samples):
    """屏幕空间 AO。"""
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= W or y >= H:
        return
    d0 = depth[y, x]
    if d0 > 9.0:
        out[y, x] = 1.0
        return
    nx = normal[y, x, 0]
    ny = normal[y, x, 1]
    occ = 0.0
    seed = (x * 1973 + y * 9277) & 0xFFFF
    for i in range(samples):
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        rx = ((seed & 0xFF) / 255.0 - 0.5) * 2.0
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        ry = ((seed & 0xFF) / 255.0 - 0.5) * 2.0
        sx = int(x + (rx + nx * 0.5) * radius / d0 * 40.0)
        sy = int(y + (ry - ny * 0.5) * radius / d0 * 40.0)
        if sx < 0 or sy < 0 or sx >= W or sy >= H:
            continue
        d1 = depth[sy, sx]
        if d1 + 0.02 < d0:
            occ += 1.0
    out[y, x] = 1.0 - occ / samples

@cuda.jit
def ssr_kernel(depth, normal, color, out, W, H):
    """极简屏幕空间反射。"""
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= W or y >= H:
        return
    out[y, x, 0] = color[y, x, 0]
    out[y, x, 1] = color[y, x, 1]
    out[y, x, 2] = color[y, x, 2]
    d0 = depth[y, x]
    if d0 > 9.0:
        return
    nx = normal[y, x, 0]
    ny = normal[y, x, 1]
    dx = nx * 8.0
    dy = -abs(ny) * 4.0 - 2.0
    px = float(x)
    py = float(y)
    for step in range(24):
        px += dx
        py += dy
        ix = int(px)
        iy = int(py)
        if ix < 0 or iy < 0 or ix >= W or iy >= H:
            break
        if depth[iy, ix] < d0 - 0.1:
            out[y, x, 0] = color[y, x, 0] * 0.6 + color[iy, ix, 0] * 0.4
            out[y, x, 1] = color[y, x, 1] * 0.6 + color[iy, ix, 1] * 0.4
            out[y, x, 2] = color[y, x, 2] * 0.6 + color[iy, ix, 2] * 0.4
            break

@cuda.jit
def bloom_tonemap_kernel(src, out, W, H, enable_bloom):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= W or y >= H:
        return
    r = src[y, x, 0]
    g = src[y, x, 1]
    b = src[y, x, 2]
    if enable_bloom != 0:
        br = 0.0
        bg = 0.0
        bb = 0.0
        cnt = 0
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                ix = clamp_i(x + dx, 0, W - 1)
                iy = clamp_i(y + dy, 0, H - 1)
                rr = src[iy, ix, 0]
                gg = src[iy, ix, 1]
                bbv = src[iy, ix, 2]
                Y = 0.2126 * rr + 0.7152 * gg + 0.0722 * bbv
                if Y > 0.8:
                    br += rr
                    bg += gg
                    bb += bbv
                    cnt += 1
        if cnt > 0:
            r += br / cnt * 0.4
            g += bg / cnt * 0.4
            b += bb / cnt * 0.4
    r = (r / (1.0 + r)) ** 0.45
    g = (g / (1.0 + g)) ** 0.45
    b = (b / (1.0 + b)) ** 0.45
    if r > 1.0:
        r = 1.0
    if g > 1.0:
        g = 1.0
    if b > 1.0:
        b = 1.0
    out[y, x, 0] = uint8(r * 255.0)
    out[y, x, 1] = uint8(g * 255.0)
    out[y, x, 2] = uint8(b * 255.0)

@cuda.jit
def fxaa_luma_kernel(src, out, W, H):
    """极简 FXAA：边缘处混合邻域。"""
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= W or y >= H:
        return
    def luma(ix, iy):
        ix = clamp_i(ix, 0, W - 1)
        iy = clamp_i(iy, 0, H - 1)
        return 0.299 * src[iy, ix, 0] + 0.587 * src[iy, ix, 1] + 0.114 * src[iy, ix, 2]

    l = luma(x, y)
    lL = luma(x - 1, y)
    lR = luma(x + 1, y)
    lU = luma(x, y - 1)
    lD = luma(x, y + 1)
    edge = abs(lL - lR) + abs(lU - lD)
    if edge > 20.0:
        for c in range(3):
            acc = (float(src[y, x, c])
                   + float(src[y, clamp_i(x - 1, 0, W - 1), c])
                   + float(src[y, clamp_i(x + 1, 0, W - 1), c])
                   + float(src[clamp_i(y - 1, 0, H - 1), x, c])
                   + float(src[clamp_i(y + 1, 0, H - 1), x, c])) / 5.0
            out[y, x, c] = uint8(acc)
    else:
        out[y, x, 0] = src[y, x, 0]
        out[y, x, 1] = src[y, x, 1]
        out[y, x, 2] = src[y, x, 2]

def main():
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    print("=" * 60)
    print("023 | 屏幕空间效果全家桶")
    print("=" * 60)
    W, H = 640, 360
    depth, normal, color = make_gbuffer(W, H)
    color[H // 3, W // 2] = [3, 3, 2]
    d_depth = cuda.to_device(depth)
    d_n = cuda.to_device(normal)
    d_c = cuda.to_device(color)
    d_ao = cuda.device_array((H, W), np.float32)
    d_ssr = cuda.device_array((H, W, 3), np.float32)
    d_ldr = cuda.device_array((H, W, 3), np.uint8)
    d_fxaa = cuda.device_array((H, W, 3), np.uint8)
    grid = ((W + 15) // 16, (H + 15) // 16)
    block = (16, 16)
    timings = {}
    t0 = time.perf_counter()
    ssao_kernel[grid, block](d_depth, d_n, d_ao, W, H, 0.5, 16)
    cuda.synchronize()
    timings["SSAO"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    ssr_kernel[grid, block](d_depth, d_n, d_c, d_ssr, W, H)
    cuda.synchronize()
    timings["SSR"] = time.perf_counter() - t0
    ssr = d_ssr.copy_to_host()
    ao = d_ao.copy_to_host()
    comp = ssr * ao[..., None]
    d_comp = cuda.to_device(comp.astype(np.float32))
    t0 = time.perf_counter()
    bloom_tonemap_kernel[grid, block](d_comp, d_ldr, W, H, 1)
    cuda.synchronize()
    timings["Bloom+TM"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    fxaa_luma_kernel[grid, block](d_ldr, d_fxaa, W, H)
    cuda.synchronize()
    timings["FXAA"] = time.perf_counter() - t0
    final = d_fxaa.copy_to_host()
    for k, v in timings.items():
        print(f"  {k:10s}: {v*1e3:7.3f} ms")
    print(f"  总计: {sum(timings.values())*1e3:.3f} ms")
    save_png(out / "ss_final.png", final)
    save_png(out / "ssao.png", (np.clip(ao, 0, 1) * 255).astype(np.uint8))
    print("✓ 屏幕空间效果管线完成。")

if __name__ == "__main__":
    main()
