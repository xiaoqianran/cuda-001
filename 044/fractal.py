#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""044 | Mandelbrot / Julia / Mandelbulb ray marching + 着色。"""
from pathlib import Path
import numpy as np
from numba import cuda
from PIL import Image
import math, time

@cuda.jit(device=True)
def clamp255(v):
    if v < 0.0:
        return 0
    if v > 255.0:
        return 255
    return int(v)

@cuda.jit
def mandelbrot_kernel(out, W, H, center_x, center_y, scale, max_iter):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= W or y >= H:
        return
    cx = center_x + (x / W - 0.5) * scale * W / H
    cy = center_y + (y / H - 0.5) * scale
    zx, zy = 0.0, 0.0
    i = 0
    while i < max_iter and zx * zx + zy * zy < 4.0:
        nzx = zx * zx - zy * zy + cx
        zy = 2.0 * zx * zy + cy
        zx = nzx
        i += 1
    if i >= max_iter:
        out[y, x, 0] = 0
        out[y, x, 1] = 0
        out[y, x, 2] = 0
    else:
        t = i / max_iter
        out[y, x, 0] = clamp255(9.0 * (1 - t) * t * t * t * 255.0)
        out[y, x, 1] = clamp255(15.0 * (1 - t) * (1 - t) * t * t * 255.0)
        out[y, x, 2] = clamp255(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255.0)

@cuda.jit
def julia_kernel(out, W, H, jx, jy, scale, max_iter):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= W or y >= H:
        return
    zx = (x / W - 0.5) * scale * W / H
    zy = (y / H - 0.5) * scale
    i = 0
    while i < max_iter and zx * zx + zy * zy < 4.0:
        nzx = zx * zx - zy * zy + jx
        zy = 2.0 * zx * zy + jy
        zx = nzx
        i += 1
    t = i / float(max_iter)
    out[y, x, 0] = clamp255(255.0 * t)
    out[y, x, 1] = clamp255(100.0 * (1.0 - t))
    out[y, x, 2] = clamp255(200.0 * (1.0 - t * 0.5))

@cuda.jit(device=True)
def mandelbulb_de(pos_x, pos_y, pos_z, power, iters):
    """距离估计 Mandelbulb。"""
    x, y, z = pos_x, pos_y, pos_z
    dr = 1.0
    r = 0.0
    for i in range(iters):
        r = math.sqrt(x * x + y * y + z * z)
        if r > 2.0:
            break
        # 钳制 z/r
        zr = z / (r + 1e-12)
        if zr < -1.0:
            zr = -1.0
        if zr > 1.0:
            zr = 1.0
        theta = math.acos(zr)
        phi = math.atan2(y, x)
        dr = (r ** (power - 1.0)) * power * dr + 1.0
        zr2 = r ** power
        theta *= power
        phi *= power
        x = zr2 * math.sin(theta) * math.cos(phi) + pos_x
        y = zr2 * math.sin(theta) * math.sin(phi) + pos_y
        z = zr2 * math.cos(theta) + pos_z
    return 0.5 * math.log(r + 1e-12) * r / dr

@cuda.jit
def mandelbulb_kernel(out, W, H, power):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= W or y >= H:
        return
    u = (x + 0.5) / W * 2 - 1
    v = 1 - (y + 0.5) / H * 2
    ox, oy, oz = 0.0, 0.0, -2.5
    dx, dy, dz = u * 0.6, v * 0.6, 1.0
    inv = 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz)
    dx *= inv
    dy *= inv
    dz *= inv
    t = 0.0
    hit = 0
    for s in range(80):
        px = ox + dx * t
        py = oy + dy * t
        pz = oz + dz * t
        d = mandelbulb_de(px, py, pz, power, 6)
        if d < 0.001:
            hit = 1
            break
        t += d
        if t > 10.0:
            break
    if hit == 0:
        out[y, x, 0] = 10
        out[y, x, 1] = 10
        out[y, x, 2] = 20
        return
    eps = 0.001
    px = ox + dx * t
    py = oy + dy * t
    pz = oz + dz * t
    nx = mandelbulb_de(px + eps, py, pz, power, 6) - mandelbulb_de(px - eps, py, pz, power, 6)
    ny = mandelbulb_de(px, py + eps, pz, power, 6) - mandelbulb_de(px, py - eps, pz, power, 6)
    nz = mandelbulb_de(px, py, pz + eps, power, 6) - mandelbulb_de(px, py, pz - eps, power, 6)
    nl = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    nx /= nl
    ny /= nl
    nz /= nl
    ao = 1.0
    for k in range(1, 6):
        step = k * 0.02
        ao -= 0.15 * (step - mandelbulb_de(px + nx * step, py + ny * step, pz + nz * step, power, 6))
    if ao < 0.0:
        ao = 0.0
    ndl = nx * 0.4 + ny * 0.7 + nz * 0.3
    if ndl < 0.0:
        ndl = 0.0
    c = (0.3 + 0.7 * ndl) * ao
    out[y, x, 0] = clamp255(c * 180.0 + 40.0)
    out[y, x, 1] = clamp255(c * 120.0 + 20.0)
    out[y, x, 2] = clamp255(c * 255.0)

def main():
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    print("=" * 60)
    print("044 | 分形渲染")
    print("=" * 60)
    W, H = 640, 480
    d_out = cuda.device_array((H, W, 3), np.uint8)
    grid = ((W + 15) // 16, (H + 15) // 16)
    mandelbrot_kernel[grid, (16, 16)](d_out, W, H, -0.5, 0.0, 3.0, 200)
    Image.fromarray(d_out.copy_to_host()).save(out / "mandelbrot.png")
    mandelbrot_kernel[grid, (16, 16)](d_out, W, H, -0.743643887037151, 0.131825904205312, 0.001, 400)
    Image.fromarray(d_out.copy_to_host()).save(out / "mandelbrot_zoom.png")
    julia_kernel[grid, (16, 16)](d_out, W, H, -0.8, 0.156, 3.0, 200)
    Image.fromarray(d_out.copy_to_host()).save(out / "julia.png")
    t0 = time.perf_counter()
    mandelbulb_kernel[grid, (16, 16)](d_out, W, H, 8.0)
    cuda.synchronize()
    print(f"Mandelbulb: {(time.perf_counter()-t0)*1000:.1f} ms")
    Image.fromarray(d_out.copy_to_host()).save(out / "mandelbulb.png")
    print("✓ 2D/3D 分形完成。")

if __name__ == "__main__":
    main()
