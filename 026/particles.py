#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""026 | GPU 粒子：位置/速度 device 数组，重力更新，软件 billboard 点splat additive 合成。"""
from pathlib import Path
import numpy as np
from numba import cuda, float32
import time
from PIL import Image

N = 100_000

@cuda.jit
def init_particles(pos, vel, life, seed0):
    i = cuda.grid(1)
    if i >= pos.shape[0]:
        return
    s = (i * 1664525 + seed0) & 0xFFFFFFFF
    # 内联随机
    s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
    r0 = (s & 0xFFFFFF) / 16777216.0
    s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
    r1 = (s & 0xFFFFFF) / 16777216.0
    s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
    r2 = (s & 0xFFFFFF) / 16777216.0
    s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
    r3 = (s & 0xFFFFFF) / 16777216.0
    s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
    r4 = (s & 0xFFFFFF) / 16777216.0
    s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
    r5 = (s & 0xFFFFFF) / 16777216.0
    s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
    r6 = (s & 0xFFFFFF) / 16777216.0
    pos[i, 0] = (r0 - 0.5) * 0.5
    pos[i, 1] = r1 * 0.2
    pos[i, 2] = (r2 - 0.5) * 0.5
    vel[i, 0] = (r3 - 0.5) * 2.0
    vel[i, 1] = r4 * 4.0 + 1.0
    vel[i, 2] = (r5 - 0.5) * 2.0
    life[i] = r6

@cuda.jit
def update_particles(pos, vel, life, dt, gravity):
    """Compute 更新：重力 + 生命周期循环。"""
    i = cuda.grid(1)
    if i >= pos.shape[0]:
        return
    life[i] -= dt
    if life[i] <= 0.0:
        s = (i * 1103515245 + 12345) & 0x7fffffff
        pos[i, 0] = ((s % 1000) / 1000.0 - 0.5) * 0.5
        pos[i, 1] = 0.0
        pos[i, 2] = (((s // 1000) % 1000) / 1000.0 - 0.5) * 0.5
        vel[i, 0] = ((s % 200) / 100.0 - 1.0)
        vel[i, 1] = 2.0 + (s % 100) / 50.0
        vel[i, 2] = (((s // 7) % 200) / 100.0 - 1.0)
        life[i] = 1.0 + (s % 100) / 100.0
    vel[i, 1] += gravity * dt
    pos[i, 0] += vel[i, 0] * dt
    pos[i, 1] += vel[i, 1] * dt
    pos[i, 2] += vel[i, 2] * dt

@cuda.jit
def render_additive(pos, life, img, W, H):
    """Billboard 点：投影到屏幕后 additive 混合。"""
    i = cuda.grid(1)
    if i >= pos.shape[0]:
        return
    z = pos[i, 2] + 3.0
    if z < 0.1:
        return
    px = int((pos[i, 0] / z * 1.5 + 0.5) * W)
    py = int((-pos[i, 1] / z * 1.5 + 0.7) * H)
    age = life[i]
    if age < 0.0:
        age = 0.0
    if age > 1.0:
        age = 1.0
    r = 1.0
    g = 0.5 * age
    b = 0.1
    rad = 2
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            x = px + dx
            y = py + dy
            if x >= 0 and x < W and y >= 0 and y < H:
                cuda.atomic.add(img, (y, x, 0), r * 8.0)
                cuda.atomic.add(img, (y, x, 1), g * 8.0)
                cuda.atomic.add(img, (y, x, 2), b * 8.0)

def main():
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    print("=" * 60)
    print(f"026 | GPU 粒子 N={N}")
    print("=" * 60)
    pos = cuda.device_array((N, 3), np.float32)
    vel = cuda.device_array((N, 3), np.float32)
    life = cuda.device_array(N, np.float32)
    threads = 256
    blocks = (N + threads - 1) // threads
    init_particles[blocks, threads](pos, vel, life, 42)
    t0 = time.perf_counter()
    for _ in range(60):
        update_particles[blocks, threads](pos, vel, life, np.float32(1 / 60), np.float32(-9.8))
    cuda.synchronize()
    print(f"60 帧更新: {(time.perf_counter()-t0)*1e3:.2f} ms")
    W, H = 640, 360
    img = cuda.to_device(np.zeros((H, W, 3), np.float32))
    t0 = time.perf_counter()
    render_additive[blocks, threads](pos, life, img, W, H)
    cuda.synchronize()
    print(f"渲染: {(time.perf_counter()-t0)*1e3:.2f} ms")
    h = img.copy_to_host()
    h = np.clip(h, 0, 255).astype(np.uint8)
    Image.fromarray(h).save(out / "particles.png")
    print("✓ 10万粒子 GPU 更新+渲染完成。")

if __name__ == "__main__":
    main()
