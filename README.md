# GPU 加速渲染 · 50 项目学习路线图

难度阶梯：Lv.1 入门 → Lv.5 专家。每个目录 `001`…`050` 含 **README + 可运行源码（简体中文注释）+ output/**。

## 快速开始

```bash
# Python / Numba 项目（示例）
cd 001 && python brightness_grayscale.py

# C++ CUDA 项目（示例）
cd 009 && make && ./simple_rt
```

依赖：`numpy`、`numba`、`pillow`；可选 `opencv`、`torch`。C++/CUDA 需 `nvcc`（CUDA 12+/13）。

无显示器时：OpenGL/Vulkan 类项目提供 **离线写图 / 软件光栅 / CUDA compute** 回退。

---

## 项目索引

### 方向一 · 图像后处理（1–8）

| # | 标题 | 栈 | 入口 |
|---|------|-----|------|
| 001 | 亮度/灰度 | Numba | `python brightness_grayscale.py` |
| 002 | 高斯模糊 shared mem | Numba | `python gaussian_blur.py` |
| 003 | Sobel 边缘 | Numba | `python sobel_edge.py` |
| 004 | Tone Mapping | Numba | `python tone_mapping.py` |
| 005 | HE + CLAHE | Numba | `python hist_clahe.py` |
| 006 | 多 Pass 管线 | Numba | `python pipeline.py` |
| 007 | Bilateral Filter | C++ CUDA | `make && ./bilateral` |
| 008 | FFT 频域滤波 | C++ cuFFT | `make && ./fft_filter` |

### 方向二 · 光线追踪（9–17）

| # | 标题 | 入口 |
|---|------|------|
| 009 | 最简单光追 | `make && ./simple_rt` |
| 010 | 多物体+阴影+反射 | `make && ./scene_rt` |
| 011 | 三角网格+BVH | `make && ./mesh_bvh` |
| 012 | AA+景深 | `make && ./aa_dof` |
| 013 | 材质系统 | `make && ./materials` |
| 014 | AO+GI 路径追踪 | `make && ./path_gi` |
| 015 | 光追优化 | `make && ./rt_opt` |
| 016 | 运动模糊+体积 | `make && ./volume_mb` |
| 017 | 小型 PBRT | `make && ./mini_pbrt` |

### 方向三 · 实时渲染（18–25）

| # | 标题 | 入口 |
|---|------|------|
| 018 | 管线立方体 Phong | `make && ./soft_gl` |
| 019 | 纹理+法线 | `make && ./texture_normal` |
| 020 | Shadow Mapping | `make && ./shadow_map` |
| 021 | PBR | `make && ./pbr` |
| 022 | 延迟渲染 | `make && ./deferred` |
| 023 | 屏幕空间效果 | `python ss_effects.py` |
| 024 | 地形 LOD | `make && ./terrain` |
| 025 | 迷你引擎 | `make && ./mini_engine` |

### 方向四 · 粒子（26–31）

| # | 标题 | 入口 |
|---|------|------|
| 026 | 粒子基础 | `python particles.py` |
| 027 | 烟花 | `python fireworks.py` |
| 028 | SPH 流体 | `python sph.py` |
| 029 | 体积火焰 | `python volume_fire.py` |
| 030 | 布料 | `python cloth.py` |
| 031 | 碎片化 | `python fracture.py` |

### 方向五 · 视频/图像生成（32–36）

| # | 标题 | 入口 |
|---|------|------|
| 032 | 视频滤镜管线 | `python video_pipeline.py` |
| 033 | 光流 | `python optical_flow.py` |
| 034 | 风格迁移 | `python style_transfer.py` |
| 035 | 超分辨率 | `python super_res.py` |
| 036 | 全景拼接 | `python panorama.py` |

### 方向六 · 科学可视化（37–41）

| # | 标题 | 入口 |
|---|------|------|
| 037 | 体渲染 | `python volume_render.py` |
| 038 | 流场 | `python flow_viz.py` |
| 039 | Marching Cubes | `python marching_cubes.py` |
| 040 | 点云 | `python point_cloud.py` |
| 041 | 地震体 | `python seismic_viz.py` |

### 方向七 · 程序化（42–45）

| # | 标题 | 入口 |
|---|------|------|
| 042 | 噪声纹理 | `python noise_tex.py` |
| 043 | 程序化地形 | `python proc_terrain.py` |
| 044 | 分形 | `python fractal.py` |
| 045 | 植被 | `python vegetation.py` |

### 方向八 · 物理耦合（46–48）

| # | 标题 | 入口 |
|---|------|------|
| 046 | 稳定流体 | `python stable_fluids.py` |
| 047 | 破碎仿真 | `python fracture_sim.py` |
| 048 | FFT 海洋 | `python ocean.py` |

### 方向九 · AI 渲染（49–50）

| # | 标题 | 入口 |
|---|------|------|
| 049 | 神经/引导降噪 | `python denoise.py` |
| 050 | NeRF | `python nerf.py` |

---

## 建议学习顺序

1. **入门冲刺** 1→2→3→9→18  
2. **图像基础** 4→5→6→7→8  
3. **光追深入** 10→11→12→13→14  
4. **实时渲染** 19→20→21→22→23  
5. **特效仿真** 26→27→42→28→46  
6. **高级/科学/AI** 其余按兴趣  

共享工具：`common/image_utils.py`、`common/rt_math.cuh`。
