#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""045 | 程序化植被：L-system 树 + GPU 实例化草 + 风摆 + LOD。"""
from pathlib import Path
import numpy as np
from numba import cuda
from PIL import Image
import math

def lsystem_tree(iters=4):
    rules = {"F": "FF+[+F-F-F]-[-F+F+F]"}
    s = "F"
    for _ in range(iters):
        s = "".join(rules.get(c, c) for c in s)
    stack = []
    x, y, ang = 0.0, 0.0, 90.0
    segs = []
    step, da = 0.08, 22.0
    for c in s:
        if c == "F":
            rad = math.radians(ang)
            nx, ny = x + step*math.cos(rad), y + step*math.sin(rad)
            segs.append((x, y, nx, ny))
            x, y = nx, ny
        elif c == "+": ang += da
        elif c == "-": ang -= da
        elif c == "[": stack.append((x, y, ang))
        elif c == "]": x, y, ang = stack.pop()
    return segs

@cuda.jit
def render_grass(pos, out, n, W, H, time, wind):
    i = cuda.grid(1)
    if i >= n: return
    z = pos[i, 2] + 3.0
    if z < 0.3: return
    stride = int(z)
    if stride < 1: stride = 1
    if (i % stride) != 0: return
    px = int((pos[i, 0]/z + 0.5) * W)
    base_y = int((-pos[i, 1]/z * 0.5 + 0.75) * H)
    sway = math.sin(time * 3.0 + pos[i, 0] * 5.0) * wind * 8
    height = int(15.0 / z)
    if height < 3: height = 3
    if height > 20: height = 20
    for h in range(height):
        xx = px + int(sway * h / height)
        yy = base_y - h
        if xx >= 0 and yy >= 0 and xx < W and yy < H:
            t = h / float(height)
            out[yy, xx, 0] = int(30 * (1-t))
            out[yy, xx, 1] = int(100 + 80 * t)
            out[yy, xx, 2] = 20

def main():
    out = Path(__file__).parent/"output"; out.mkdir(exist_ok=True)
    print("="*60); print("045 | 植被与森林"); print("="*60)
    segs = lsystem_tree(4)
    print(f"L-system 树段数: {len(segs)}")
    W, H = 512, 512
    img = np.zeros((H,W,3), np.uint8)
    img[:,:] = [135, 206, 235]
    img[H//2:,:] = [60, 120, 40]
    for x0,y0,x1,y1 in segs:
        for t in range(20):
            x = int(W*0.5 + (x0+(x1-x0)*t/20)*120)
            y = int(H*0.85 - (y0+(y1-y0)*t/20)*120)
            if 0<=x<W and 0<=y<H: img[y,x]=[80,50,20]
    Image.fromarray(img).save(out/"tree_lsystem.png")
    n_grass = 80_000
    rng = np.random.default_rng(0)
    pos = np.zeros((n_grass,3), np.float32)
    pos[:,0] = rng.uniform(-2,2,n_grass)
    pos[:,2] = rng.uniform(-2,2,n_grass)
    d_pos = cuda.to_device(pos)
    h = np.zeros((360,640,3), np.uint8)
    h[:] = [100,150,220]
    h[200:] = [50,100,30]
    d_out = cuda.to_device(h)
    render_grass[(n_grass+255)//256, 256](d_pos, d_out, n_grass, 640, 360, 0.5, 1.0)
    Image.fromarray(d_out.copy_to_host()).save(out/"grass.png")
    print(f"草实例: {n_grass}")
    print("✓ 植被系统完成。")

if __name__=="__main__":
    main()
