#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""048 | Phillips 频谱 FFT 波浪 + 海面 PBR 着色。"""
from pathlib import Path
import numpy as np
from numba import cuda
from PIL import Image
import math, time

def phillips(kx, kz, wind, A=0.0005, g=9.81):
    k2 = kx*kx + kz*kz
    if k2 < 1e-12: return 0.0
    k = math.sqrt(k2)
    L = wind*wind/g
    kdotw = (kx*1.0 + kz*0.0)/k
    if kdotw < 0: kdotw = 0
    return A * math.exp(-1.0/(k*L)**2) / (k2*k2) * (kdotw**2) * math.exp(-k2*(L*0.001)**2)

def generate_spectrum(N, L, wind, seed=0):
    rng = np.random.default_rng(seed)
    h0 = np.zeros((N,N), dtype=np.complex64)
    for n in range(N):
        for m in range(N):
            kx = 2*math.pi*(n-N/2)/L
            kz = 2*math.pi*(m-N/2)/L
            p = phillips(kx,kz,wind)
            er,ei = rng.normal(), rng.normal()
            h0[n,m] = math.sqrt(p/2)*(er+1j*ei)
    return h0

def ocean_height_field(h0, N, L, t, g=9.81):
    h_t = np.zeros((N,N), dtype=np.complex64)
    for n in range(N):
        for m in range(N):
            kx = 2*math.pi*(n-N/2)/L
            kz = 2*math.pi*(m-N/2)/L
            k = math.sqrt(kx*kx+kz*kz)
            w = math.sqrt(g*k) if k>0 else 0
            h_t[n,m] = h0[n,m]*np.exp(1j*w*t) + np.conj(h0[(N-n)%N,(N-m)%N])*np.exp(-1j*w*t)
    height = np.fft.ifft2(np.fft.ifftshift(h_t)).real.astype(np.float32)
    dx = np.gradient(height, axis=0).astype(np.float32)
    dz = np.gradient(height, axis=1).astype(np.float32)
    return height, dx, dz

@cuda.jit(device=True)
def clamp01(v):
    if v < 0.0: return 0.0
    if v > 1.0: return 1.0
    return v

@cuda.jit
def shade_ocean(height, dx, dz, out, W, H, N):
    x = cuda.blockIdx.x*cuda.blockDim.x+cuda.threadIdx.x
    y = cuda.blockIdx.y*cuda.blockDim.y+cuda.threadIdx.y
    if x>=W or y>=H: return
    u=(x+0.5)/W*2-1; v=1-(y+0.5)/H*2
    ox,oy,oz=0.0,1.2,0.0
    dxr,dyr,dzr=u*0.9, v*0.5-0.45, 1.0
    inv=1.0/math.sqrt(dxr*dxr+dyr*dyr+dzr*dzr)
    dxr*=inv; dyr*=inv; dzr*=inv
    if abs(dyr)<1e-5:
        out[y,x,0]=20; out[y,x,1]=30; out[y,x,2]=50; return
    t=-oy/dyr
    if t<0:
        out[y,x,0]=int(100+v*50); out[y,x,1]=int(150+v*40); out[y,x,2]=220; return
    px=ox+dxr*t; pz=oz+dzr*t
    fx=(px*0.15+0.5)*N; fz=(pz*0.15+0.5)*N
    ix=int(fx)%N; iz=int(fz)%N
    if ix<0: ix+=N
    if iz<0: iz+=N
    h=height[iz,ix]*0.3
    sx=dx[iz,ix]; sz=dz[iz,ix]
    nx,ny,nz=-sx,1.0,-sz
    nl=math.sqrt(nx*nx+ny*ny+nz*nz); nx/=nl; ny/=nl; nz/=nl
    Vdx,Vdy,Vdz=-dxr,-dyr,-dzr
    ndv=nx*Vdx+ny*Vdy+nz*Vdz
    if ndv<0: ndv=0.0
    F=0.02+0.98*(1-ndv)**5
    water_r=0.02; water_g=0.15; water_b=0.25
    sky_r,sky_g,sky_b=0.4,0.6,0.9
    steep=sx*sx+sz*sz
    foam=steep*8.0
    if foam>1.0: foam=1.0
    r=water_r*(1-F)+sky_r*F
    g=water_g*(1-F)+sky_g*F
    b=water_b*(1-F)+sky_b*F
    r=r*(1-foam)+foam; g=g*(1-foam)+foam; b=b*(1-foam)+foam
    Lx,Ly,Lz=0.5,0.7,0.3
    Hx,Hy,Hz=Lx+Vdx,Ly+Vdy,Lz+Vdz
    hl=math.sqrt(Hx*Hx+Hy*Hy+Hz*Hz)+1e-8
    ndh=(nx*Hx+ny*Hy+nz*Hz)/hl
    if ndh<0: ndh=0.0
    spec=ndh**128 * F
    r+=spec; g+=spec; b+=spec
    r=clamp01(r); g=clamp01(g); b=clamp01(b)
    out[y,x,0]=int(r*255); out[y,x,1]=int(g*255); out[y,x,2]=int(b*255)

def main():
    out=Path(__file__).parent/"output"; out.mkdir(exist_ok=True)
    print("="*60); print("048 | FFT 海洋"); print("="*60)
    N,L=64,100.0
    h0=generate_spectrum(N,L,32.0)
    t0=time.perf_counter()
    height,dx,dz=ocean_height_field(h0,N,L,2.0)
    print(f"频谱+IFFT: {(time.perf_counter()-t0)*1000:.1f} ms")
    d_h=cuda.to_device(height); d_dx=cuda.to_device(dx); d_dz=cuda.to_device(dz)
    W,H=640,360
    d_out=cuda.device_array((H,W,3),np.uint8)
    shade_ocean[((W+15)//16,(H+15)//16),(16,16)](d_h,d_dx,d_dz,d_out,W,H,N)
    Image.fromarray(d_out.copy_to_host()).save(out/"ocean.png")
    hvis=(height-height.min())/(height.max()-height.min()+1e-6)
    Image.fromarray((hvis*255).astype(np.uint8)).save(out/"height.png")
    print("✓ 海洋 FFT 波浪 + PBR 着色完成。")

if __name__=="__main__":
    main()
