#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""042 | Perlin / Worley / fBM → 木纹、大理石、火焰纹理。"""
from pathlib import Path
import numpy as np
from numba import cuda
from PIL import Image
import math, time

@cuda.jit(device=True)
def fade(t):
    return t*t*t*(t*(t*6-15)+10)

@cuda.jit(device=True)
def lerp(a,b,t):
    return a+t*(b-a)

@cuda.jit(device=True)
def grad2(h,x,y):
    hh = h & 3
    if hh==0: return x+y
    if hh==1: return -x+y
    if hh==2: return x-y
    return -x-y

@cuda.jit(device=True)
def perlin2(x,y,perm):
    xi = int(math.floor(x)) & 255
    yi = int(math.floor(y)) & 255
    xf = x-math.floor(x); yf=y-math.floor(y)
    u,v = fade(xf), fade(yf)
    aa=perm[perm[xi]+yi]; ab=perm[perm[xi]+yi+1]
    ba=perm[perm[xi+1]+yi]; bb=perm[perm[xi+1]+yi+1]
    x1=lerp(grad2(aa,xf,yf), grad2(ba,xf-1,yf), u)
    x2=lerp(grad2(ab,xf,yf-1), grad2(bb,xf-1,yf-1), u)
    return lerp(x1,x2,v)

@cuda.jit(device=True)
def fbm(x,y,perm,octaves):
    amp=1.0; freq=1.0; s=0.0; n=0.0
    for i in range(octaves):
        s += amp*perlin2(x*freq,y*freq,perm)
        n += amp; amp*=0.5; freq*=2.0
    return s/n

@cuda.jit(device=True)
def worley2(x,y,seed):
    xi=int(math.floor(x)); yi=int(math.floor(y))
    dmin=1e30
    for dy in range(-1,2):
        for dx in range(-1,2):
            cx,cy=xi+dx,yi+dy
            n=(cx*374761393+cy*668265263+seed)&0x7fffffff
            fx=cx+(n%1000)/1000.0
            n=(n*1664525+1013904223)&0x7fffffff
            fy=cy+(n%1000)/1000.0
            d=(fx-x)**2+(fy-y)**2
            if d<dmin: dmin=d
    return math.sqrt(dmin)

@cuda.jit
def gen_textures(out_wood,out_marble,out_fire,out_noise,W,H,perm,freq,octaves):
    x=cuda.blockIdx.x*cuda.blockDim.x+cuda.threadIdx.x
    y=cuda.blockIdx.y*cuda.blockDim.y+cuda.threadIdx.y
    if x>=W or y>=H: return
    u=x/float(W); v=y/float(H)
    n=fbm(u*freq,v*freq,perm,octaves)
    n01=n*0.5+0.5
    out_noise[y,x]=n01*255
    rings=math.sin((u*freq*0.3+n*0.4)*20.0)*0.5+0.5
    out_wood[y,x,0]=(0.4+0.3*rings)*255
    out_wood[y,x,1]=(0.2+0.15*rings)*255
    out_wood[y,x,2]=(0.08+0.05*rings)*255
    m=abs(math.sin((u*3+n*2)*3.14159))
    out_marble[y,x,0]=(0.85+0.1*m)*255
    out_marble[y,x,1]=(0.85+0.1*m)*255
    out_marble[y,x,2]=(0.9+0.05*m)*255
    w=worley2(u*6,v*4,42)
    f=fbm(u*4,v*3+0.5,perm,4)*0.5+0.5
    flame=(1.0-v)*(1.0-w)*f*2.0
    if flame<0: flame=0
    if flame>1: flame=1
    out_fire[y,x,0]=flame*255*1.2 if flame*255*1.2<255 else 255
    out_fire[y,x,1]=flame*flame*200
    out_fire[y,x,2]=flame*flame*flame*80

def main():
    out=Path(__file__).parent/"output"; out.mkdir(exist_ok=True)
    print("="*60); print("042 | 程序化噪声与纹理"); print("="*60)
    rng=np.random.default_rng(0)
    p=np.arange(256,dtype=np.int32); rng.shuffle(p)
    perm=np.concatenate([p,p]).astype(np.int32)
    d_perm=cuda.to_device(perm)
    W,H=512,512
    d_wood=cuda.device_array((H,W,3),np.float32)
    d_mar=cuda.device_array((H,W,3),np.float32)
    d_fire=cuda.device_array((H,W,3),np.float32)
    d_n=cuda.device_array((H,W),np.float32)
    grid=((W+15)//16,(H+15)//16)
    t0=time.perf_counter()
    gen_textures[grid,(16,16)](d_wood,d_mar,d_fire,d_n,W,H,d_perm,8.0,6)
    cuda.synchronize()
    print(f"{W}x{H} 生成: {(time.perf_counter()-t0)*1000:.1f} ms")
    for name,arr in [("wood",d_wood),("marble",d_mar),("fire",d_fire)]:
        Image.fromarray(np.clip(arr.copy_to_host(),0,255).astype(np.uint8)).save(out/f"{name}.png")
    Image.fromarray(np.clip(d_n.copy_to_host(),0,255).astype(np.uint8)).save(out/"fbm.png")
    print("✓ Perlin/fBM/Worley 纹理完成。")

if __name__=="__main__":
    main()
