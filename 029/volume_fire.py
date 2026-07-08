#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""029 | 体积火焰：粒子温度/密度 + ray march 累积 + 黑体色。"""
from pathlib import Path
import numpy as np
from numba import cuda
from PIL import Image
import math

@cuda.jit(device=True)
def blackbody(T, out_rgb):
    t = T / 4000.0
    if t < 0.2: t = 0.2
    if t > 1.0: t = 1.0
    r = t * 1.5
    if r > 1.0: r = 1.0
    g = (t - 0.2) * 1.4
    if g < 0.0: g = 0.0
    if g > 1.0: g = 1.0
    b = (t - 0.5) * 2.0
    if b < 0.0: b = 0.0
    if b > 1.0: b = 1.0
    out_rgb[0] = r
    out_rgb[1] = g
    out_rgb[2] = b

@cuda.jit
def raymarch_fire(img, W, H, particles, n, dens_scale):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= W or y >= H:
        return
    u = (x + 0.5) / W * 2 - 1
    v = 1 - (y + 0.5) / H * 2
    ox, oy, oz = 0.0, 0.5, 2.0
    dx, dy, dz = u * 0.6, v * 0.5, -1.0
    inv = 1.0 / math.sqrt(dx*dx + dy*dy + dz*dz)
    dx *= inv; dy *= inv; dz *= inv
    acc_r = acc_g = acc_b = 0.0
    transmittance = 1.0
    rgb = cuda.local.array(3, np.float32)
    for step in range(48):
        t = 0.5 + step * 0.05
        px = ox + dx * t
        py = oy + dy * t
        pz = oz + dz * t
        if py < 0:
            ground = 0.15
            acc_r += transmittance * ground
            acc_g += transmittance * ground
            acc_b += transmittance * ground * 1.1
            break
        dens = 0.0
        temp = 0.0
        for i in range(n):
            qx = px - particles[i, 0]
            qy = py - particles[i, 1]
            qz = pz - particles[i, 2]
            r2 = qx*qx + qy*qy + qz*qz
            rad = particles[i, 5]
            if r2 < rad * rad:
                rr = math.sqrt(r2)
                ww = 1.0 - rr / rad
                dens += particles[i, 4] * ww * ww
                temp += particles[i, 3] * ww * ww
        dens *= dens_scale
        if dens > 1e-4:
            temp = temp / (dens / dens_scale + 1e-6)
            blackbody(temp, rgb)
            absorb = 1.0 - math.exp(-dens * 0.15)
            acc_r += transmittance * absorb * rgb[0]
            acc_g += transmittance * absorb * rgb[1]
            acc_b += transmittance * absorb * rgb[2]
            transmittance *= (1.0 - absorb)
            if transmittance < 0.01:
                break
    acc_r += transmittance * 0.05
    acc_g += transmittance * 0.06
    acc_b += transmittance * 0.1
    if acc_r > 1.0: acc_r = 1.0
    if acc_g > 1.0: acc_g = 1.0
    if acc_b > 1.0: acc_b = 1.0
    img[y, x, 0] = acc_r * 255
    img[y, x, 1] = acc_g * 255
    img[y, x, 2] = acc_b * 255

def main():
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    print("=" * 60)
    print("029 | 体积火焰 ray march")
    print("=" * 60)
    rng = np.random.default_rng(0)
    n = 60
    parts = np.zeros((n, 6), np.float32)
    for i in range(n):
        parts[i, 0] = rng.normal(0, 0.15)
        parts[i, 1] = rng.uniform(0.1, 1.2)
        parts[i, 2] = rng.normal(0, 0.15)
        parts[i, 3] = 1500 + parts[i, 1] * 800
        parts[i, 4] = 1.5 * (1.2 - parts[i, 1])
        parts[i, 5] = 0.15 + 0.1 * rng.random()
    W, H = 400, 300
    d_p = cuda.to_device(parts)
    d_img = cuda.device_array((H, W, 3), np.float32)
    raymarch_fire[((W+15)//16, (H+15)//16), (16, 16)](d_img, W, H, d_p, n, 1.0)
    cuda.synchronize()
    img = np.clip(d_img.copy_to_host(), 0, 255).astype(np.uint8)
    Image.fromarray(img).save(out / "fire.png")
    print("✓ 体积火焰完成。")

if __name__ == "__main__":
    main()
