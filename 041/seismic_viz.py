#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""041 | 地震体：分块流式体渲染 + 切片 + 多属性。"""
from pathlib import Path
import numpy as np
from numba import cuda
from PIL import Image
import math

@cuda.jit(device=True)
def to_u8(v):
    if v < 0: return 0
    if v > 255: return 255
    return int(v)

@cuda.jit
def render_seismic_slice(amp, out, W, H, axis, slice_idx, N):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= W or y >= H: return
    if axis == 0:
        ix, iy, iz = slice_idx, int(y/H*N), int(x/W*N)
    elif axis == 1:
        iy, ix, iz = slice_idx, int(x/W*N), int(y/H*N)
    else:
        iz, ix, iy = slice_idx, int(x/W*N), int(y/H*N)
    if ix < 0: ix = 0
    if iy < 0: iy = 0
    if iz < 0: iz = 0
    if ix >= N: ix = N-1
    if iy >= N: iy = N-1
    if iz >= N: iz = N-1
    v = amp[iz, iy, ix]
    if v < 0:
        t = v
        if t < -1: t = -1.0
        out[y,x,0]=to_u8((1+t)*255); out[y,x,1]=to_u8((1+t)*255); out[y,x,2]=255
    else:
        t = v
        if t > 1: t = 1.0
        out[y,x,0]=255; out[y,x,1]=to_u8((1-t)*255); out[y,x,2]=to_u8((1-t)*255)

@cuda.jit
def volume_amp_render(amp, coh, out, W, H, N, clip_x):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= W or y >= H: return
    u=(x+0.5)/W*2-1; v=1-(y+0.5)/H*2
    ox,oy,oz=0.0,0.0,-1.5
    dx,dy,dz=u*0.6,v*0.6,1.0
    inv=1.0/math.sqrt(dx*dx+dy*dy+dz*dz)
    dx*=inv; dy*=inv; dz*=inv
    ar=ag=ab=0.0; T=1.0
    for s in range(100):
        t=0.5+s*0.018
        px,py,pz=ox+dx*t,oy+dy*t,oz+dz*t
        qx,qy,qz=px*0.4+0.5,py*0.4+0.5,pz*0.4+0.5
        if qx < clip_x or qx>1 or qy<0 or qy>1 or qz<0 or qz>1: continue
        ix=int(qx*(N-1)); iy=int(qy*(N-1)); iz=int(qz*(N-1))
        a=amp[iz,iy,ix]; c=coh[iz,iy,ix]
        alpha = abs(a)*0.3*c
        if alpha > 0.4: alpha = 0.4
        if a > 0:
            cr,cg,cb = 1.0, 1.0-a, 1.0-a
        else:
            cr,cg,cb = 1.0+a, 1.0+a, 1.0
        ar+=T*alpha*cr; ag+=T*alpha*cg; ab+=T*alpha*cb
        T*=(1-alpha)
        if T < 0.02: break
    ar+=T*0.1; ag+=T*0.1; ab+=T*0.12
    out[y,x,0]=to_u8(ar*255); out[y,x,1]=to_u8(ag*255); out[y,x,2]=to_u8(ab*255)

def make_seismic_chunk(N=40):
    zz,yy,xx=np.mgrid[0:N,0:N,0:N].astype(np.float32)
    amp = np.sin(zz*0.4)*np.exp(-((xx-N/2)**2+(yy-N/2)**2)/(2*(N*0.3)**2))
    amp += 0.3*np.sin(xx*0.3+zz*0.2)
    amp = amp.astype(np.float32)
    amp /= (np.abs(amp).max()+1e-6)
    coh = (0.5+0.5*np.cos(zz*0.2)).astype(np.float32)
    return amp, coh

def main():
    out=Path(__file__).parent/"output"; out.mkdir(exist_ok=True)
    print("="*60); print("041 | 地震体分块可视化"); print("="*60)
    N=40; W,H=400,300
    composite=np.zeros((H,W,3), np.float32)
    for chunk_id in range(3):
        amp,coh=make_seismic_chunk(N)
        amp=np.roll(amp, chunk_id*5, axis=0)
        d_amp=cuda.to_device(amp); d_coh=cuda.to_device(coh)
        d_out=cuda.device_array((H,W,3), np.float32)
        volume_amp_render[((W+15)//16,(H+15)//16),(16,16)](d_amp,d_coh,d_out,W,H,N,0.15*chunk_id)
        composite += d_out.copy_to_host()/3
        print(f"  chunk {chunk_id} 已流式渲染")
    Image.fromarray(np.clip(composite,0,255).astype(np.uint8)).save(out/"seismic_volume.png")
    amp,coh=make_seismic_chunk(N)
    d_amp=cuda.to_device(amp)
    d_sl=cuda.device_array((H,W,3), np.uint8)
    render_seismic_slice[((W+15)//16,(H+15)//16),(16,16)](d_amp,d_sl,W,H,2,N//2,N)
    Image.fromarray(d_sl.copy_to_host()).save(out/"seismic_slice.png")
    print("✓ 分块体渲染 + 切片完成。")

if __name__=="__main__":
    main()
