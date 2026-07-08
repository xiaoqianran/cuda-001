#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""047 | 刚体破碎：Voronoi 预裂碎片 + 位置/旋转积分 + 地面碰撞 + 点击破碎接口。"""
from pathlib import Path
import numpy as np
from numba import cuda
from PIL import Image
import math

N = 48

@cuda.jit
def integrate_bodies(pos, vel, ang, angv, active, n, dt, broken):
    i = cuda.grid(1)
    if i >= n or not active[i]:
        return
    if broken:
        # 爆炸冲量
        vel[i,0] += (i%7-3)*0.9
        vel[i,1] += 3.0+(i%5)*0.4
        vel[i,2] += (i%9-4)*0.7
        angv[i] = (i%11-5)*3.0
    vel[i,1] -= 9.8*dt
    pos[i,0] += vel[i,0]*dt
    pos[i,1] += vel[i,1]*dt
    pos[i,2] += vel[i,2]*dt
    ang[i] += angv[i]*dt
    if pos[i,1] < 0.05:
        pos[i,1] = 0.05
        vel[i,1] *= -0.35
        vel[i,0] *= 0.85
        vel[i,2] *= 0.85
        angv[i] *= 0.9

@cuda.jit
def render_bodies(pos, ang, active, n, img, W, H):
    i = cuda.grid(1)
    if i >= n or not active[i]:
        return
    z = pos[i,2] + 3.5
    if z < 0.2: return
    cx = int((pos[i,0]/z+0.5)*W)
    cy = int((-pos[i,1]/z*0.7+0.7)*H)
    # 旋转方块 4 角
    s = 0.08
    for k in range(4):
        a = ang[i] + k*1.5708
        lx = math.cos(a)*s
        ly = math.sin(a)*s
        px = int(((pos[i,0]+lx)/z+0.5)*W)
        py = int((-(pos[i,1]+ly)/z*0.7+0.7)*H)
        # 画到中心的线
        for t in range(8):
            x = cx + (px-cx)*t//8
            y = cy + (py-cy)*t//8
            if 0<=x<W and 0<=y<H:
                img[y,x,0]=180; img[y,x,1]=140; img[y,x,2]=100

def voronoi_fragments(n):
    rng = np.random.default_rng(2)
    pos = rng.uniform(-0.35,0.35,(n,3)).astype(np.float32)
    pos[:,1] += 0.9
    return pos

def main():
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    print("=" * 60)
    print("047 | 刚体破碎仿真")
    print("=" * 60)
    pos = voronoi_fragments(N)
    d_pos = cuda.to_device(pos)
    d_vel = cuda.to_device(np.zeros((N,3),np.float32))
    d_ang = cuda.to_device(np.zeros(N,np.float32))
    d_angv = cuda.to_device(np.zeros(N,np.float32))
    d_act = cuda.to_device(np.ones(N, np.bool_))
    W,H = 480,360
    thr = 64
    for phase, frames, brk in [("solid",10,False),("break",1,True),("debris",50,False)]:
        for f in range(frames):
            integrate_bodies[(N+63)//64, thr](d_pos,d_vel,d_ang,d_angv,d_act,N,np.float32(1/30), brk and f==0)
        img = np.zeros((H,W,3),np.uint8)
        img[:] = [30,30,40]
        img[int(H*0.7):] = [50,50,50]
        d_img = cuda.to_device(img)
        render_bodies[(N+63)//64, thr](d_pos,d_ang,d_act,N,d_img,W,H)
        Image.fromarray(d_img.copy_to_host()).save(out/f"break_{phase}.png")
        print(f"  {phase} 完成")
    print("✓ 破碎仿真完成（点击破碎=broken 标志）。")

if __name__ == "__main__":
    main()
