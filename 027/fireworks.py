#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""027 | 烟花：母粒子上升 → 爆炸子粒子；GPU emit/update/render。"""
from pathlib import Path
import numpy as np
from numba import cuda
import time
from PIL import Image

MAX_P = 100_000

@cuda.jit
def update_fw(pos, vel, life, kind, color, n, dt, frame):
    i = cuda.grid(1)
    if i >= n or kind[i] == 0:
        return
    life[i] -= dt
    g = 9.8 * dt * (0.3 if kind[i] == 1 else 1.0)
    vel[i, 1] -= g
    vel[i, 0] *= 0.995
    vel[i, 2] *= 0.995
    pos[i, 0] += vel[i, 0] * dt
    pos[i, 1] += vel[i, 1] * dt
    pos[i, 2] += vel[i, 2] * dt
    t = life[i]
    if t < 0.0: t = 0.0
    if t > 1.0: t = 1.0
    if kind[i] == 1:
        color[i, 0], color[i, 1], color[i, 2] = 1.0, 0.9, 0.5
    else:
        color[i, 0] = 1.0
        color[i, 1] = 0.3 + 0.5 * t
        color[i, 2] = 0.1 + 0.6 * (1.0 - t)
    if kind[i] == 1 and (life[i] <= 0 or (vel[i, 1] < 0 and pos[i, 1] > 2.0)):
        kind[i] = 0
        base = (i * 32) % n
        for k in range(32):
            j = (base + k) % n
            if kind[j] == 0:
                kind[j] = 2
                life[j] = 1.2
                pos[j, 0] = pos[i, 0]
                pos[j, 1] = pos[i, 1]
                pos[j, 2] = pos[i, 2]
                ang = k / 32.0 * 6.28318
                elev = (k % 8) / 8.0 * 3.14159
                sp = 3.0 + (k % 5) * 0.4
                vel[j, 0] = sp * np.sin(elev) * np.cos(ang)
                vel[j, 1] = sp * np.cos(elev)
                vel[j, 2] = sp * np.sin(elev) * np.sin(ang)
                break
    if life[i] <= 0:
        kind[i] = 0

@cuda.jit
def spawn_mothers(pos, vel, life, kind, n, frame):
    if frame % 15 != 0:
        return
    if cuda.grid(1) != 0:
        return
    i = (frame * 7) % n
    if kind[i] == 0:
        kind[i] = 1
        life[i] = 2.0
        pos[i, 0] = ((frame * 13) % 100) / 100.0 * 2 - 1
        pos[i, 1] = 0.0
        pos[i, 2] = ((frame * 17) % 100) / 100.0 * 2 - 1
        vel[i, 0] = 0.0
        vel[i, 1] = 8.0 + (frame % 5)
        vel[i, 2] = 0.0

@cuda.jit
def render(pos, life, kind, color, img, W, H):
    i = cuda.grid(1)
    if i >= pos.shape[0] or kind[i] == 0:
        return
    z = pos[i, 2] + 4.0
    if z < 0.2:
        return
    px = int((pos[i, 0] / z + 0.5) * W)
    py = int((-pos[i, 1] / z * 0.8 + 0.85) * H)
    a = life[i]
    if a < 0.0: a = 0.0
    if a > 1.0: a = 1.0
    rad = 1 if kind[i] == 2 else 2
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            x = px + dx
            y = py + dy
            if x >= 0 and x < W and y >= 0 and y < H:
                cuda.atomic.add(img, (y, x, 0), color[i, 0] * a * 12)
                cuda.atomic.add(img, (y, x, 1), color[i, 1] * a * 12)
                cuda.atomic.add(img, (y, x, 2), color[i, 2] * a * 12)

def main():
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    print("=" * 60)
    print(f"027 | 烟花粒子 MAX={MAX_P}")
    print("=" * 60)
    pos = cuda.to_device(np.zeros((MAX_P, 3), np.float32))
    vel = cuda.to_device(np.zeros((MAX_P, 3), np.float32))
    life = cuda.to_device(np.zeros(MAX_P, np.float32))
    kind = cuda.to_device(np.zeros(MAX_P, np.int32))
    color = cuda.to_device(np.zeros((MAX_P, 3), np.float32))
    thr, blk = 256, (MAX_P + 255) // 256
    W, H = 640, 360
    t0 = time.perf_counter()
    for f in range(90):
        spawn_mothers[1, 1](pos, vel, life, kind, MAX_P, f)
        update_fw[blk, thr](pos, vel, life, kind, color, MAX_P, np.float32(1/30), f)
    cuda.synchronize()
    print(f"90 帧模拟: {(time.perf_counter()-t0)*1e3:.1f} ms")
    img = cuda.to_device(np.zeros((H, W, 3), np.float32))
    render[blk, thr](pos, life, kind, color, img, W, H)
    cuda.synchronize()
    h = np.clip(img.copy_to_host(), 0, 255).astype(np.uint8)
    Image.fromarray(h).save(out / "fireworks.png")
    active = int((kind.copy_to_host() > 0).sum())
    print(f"活跃粒子: {active}")
    print("✓ 烟花 emit/update/render 完成。")

if __name__ == "__main__":
    main()
