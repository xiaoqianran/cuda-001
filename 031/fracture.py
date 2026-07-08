#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""031 | Voronoi 碎片化：静止→撞击→碎片飞散→灰尘。"""
from pathlib import Path
import numpy as np
from numba import cuda
from PIL import Image
import time

N_FRAG = 64
N_DUST = 2000

@cuda.jit
def update_frags(pos, vel, ang, angv, active, n, dt, impact):
    i = cuda.grid(1)
    if i >= n or not active[i]:
        return
    if impact:
        # 撞击瞬间给速度
        vel[i, 0] += (i % 7 - 3) * 0.8
        vel[i, 1] += 2.0 + (i % 5) * 0.5
        vel[i, 2] += (i % 9 - 4) * 0.6
        angv[i] = (i % 11 - 5) * 2.0
    vel[i, 1] -= 9.8 * dt
    pos[i, 0] += vel[i, 0] * dt
    pos[i, 1] += vel[i, 1] * dt
    pos[i, 2] += vel[i, 2] * dt
    ang[i] += angv[i] * dt
    # 碰地
    if pos[i, 1] < 0:
        pos[i, 1] = 0
        vel[i, 1] *= -0.3
        vel[i, 0] *= 0.8
        vel[i, 2] *= 0.8

@cuda.jit
def update_dust(pos, vel, life, n, dt, emit_from, n_frag, do_emit):
    i = cuda.grid(1)
    if i >= n:
        return
    if do_emit and life[i] <= 0 and i < n_frag * 10:
        f = i % n_frag
        pos[i, 0] = emit_from[f, 0]
        pos[i, 1] = emit_from[f, 1]
        pos[i, 2] = emit_from[f, 2]
        vel[i, 0] = (i % 5 - 2) * 0.5
        vel[i, 1] = (i % 3) * 0.3
        vel[i, 2] = (i % 7 - 3) * 0.5
        life[i] = 0.8
    if life[i] <= 0:
        return
    life[i] -= dt
    vel[i, 1] -= 2.0 * dt
    pos[i, 0] += vel[i, 0] * dt
    pos[i, 1] += vel[i, 1] * dt
    pos[i, 2] += vel[i, 2] * dt

@cuda.jit
def render(pos_f, n_f, pos_d, life_d, n_d, img, W, H):
    i = cuda.grid(1)
    def plot(px, py, r, g, b, s):
        for dy in range(-s, s+1):
            for dx in range(-s, s+1):
                x, y = px+dx, py+dy
                if 0 <= x < W and 0 <= y < H:
                    cuda.atomic.add(img, (y, x, 0), r)
                    cuda.atomic.add(img, (y, x, 1), g)
                    cuda.atomic.add(img, (y, x, 2), b)
    if i < n_f:
        z = pos_f[i, 2] + 3
        if z > 0.2:
            px = int((pos_f[i, 0]/z + 0.5)*W)
            py = int((-pos_f[i, 1]/z*0.8 + 0.7)*H)
            plot(px, py, 40, 30, 25, 3)
    if i < n_d and life_d[i] > 0:
        z = pos_d[i, 2] + 3
        if z > 0.2:
            px = int((pos_d[i, 0]/z + 0.5)*W)
            py = int((-pos_d[i, 1]/z*0.8 + 0.7)*H)
            a = life_d[i]
            plot(px, py, 20*a, 18*a, 15*a, 1)

def voronoi_sites(n):
    """预计算 Voronoi 种子作为碎片中心。"""
    rng = np.random.default_rng(1)
    return rng.uniform(-0.4, 0.4, (n, 3)).astype(np.float32)

def main():
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    print("=" * 60)
    print("031 | Voronoi 碎片化撞击")
    print("=" * 60)
    sites = voronoi_sites(N_FRAG)
    sites[:, 1] += 0.8  # 抬高
    pos_f = cuda.to_device(sites.copy())
    vel_f = cuda.to_device(np.zeros((N_FRAG, 3), np.float32))
    ang = cuda.to_device(np.zeros(N_FRAG, np.float32))
    angv = cuda.to_device(np.zeros(N_FRAG, np.float32))
    active = cuda.to_device(np.ones(N_FRAG, np.bool_))
    pos_d = cuda.to_device(np.zeros((N_DUST, 3), np.float32))
    vel_d = cuda.to_device(np.zeros((N_DUST, 3), np.float32))
    life_d = cuda.to_device(np.zeros(N_DUST, np.float32))
    W, H = 480, 360
    thr = 128
    # 动画序列：静止 10 帧 → 撞击 → 飞散
    for phase, frames, impact, dust in [
        ("rest", 5, False, False),
        ("impact", 1, True, True),
        ("fly", 40, False, True),
    ]:
        for f in range(frames):
            update_frags[(N_FRAG+127)//128, thr](pos_f, vel_f, ang, angv, active, N_FRAG, np.float32(1/30), impact and f==0)
            update_dust[(N_DUST+127)//128, thr](pos_d, vel_d, life_d, N_DUST, np.float32(1/30), pos_f, N_FRAG, dust and f < 5)
        img = cuda.to_device(np.zeros((H, W, 3), np.float32))
        render[(max(N_FRAG,N_DUST)+127)//128, thr](pos_f, N_FRAG, pos_d, life_d, N_DUST, img, W, H)
        cuda.synchronize()
        h = np.clip(img.copy_to_host(), 0, 255).astype(np.uint8)
        Image.fromarray(h).save(out / f"fracture_{phase}.png")
        print(f"  阶段 {phase} 已保存")
    print("✓ 破碎动画序列完成。")

if __name__ == "__main__":
    main()
