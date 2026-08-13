# -*- coding: utf-8 -*-
"""50 个实验的元数据：Modal 跑手、ingest、Pages 共用。"""

from __future__ import annotations

CATEGORIES = [
    {"id": "post", "label": "图像后处理", "range": "001–008"},
    {"id": "rt", "label": "光线追踪", "range": "009–017"},
    {"id": "realtime", "label": "实时渲染", "range": "018–025"},
    {"id": "particles", "label": "粒子仿真", "range": "026–031"},
    {"id": "video", "label": "视频 / 图像生成", "range": "032–036"},
    {"id": "sci", "label": "科学可视化", "range": "037–041"},
    {"id": "proc", "label": "程序化", "range": "042–045"},
    {"id": "phys", "label": "物理耦合", "range": "046–048"},
    {"id": "ai", "label": "AI 渲染", "range": "049–050"},
]

# kind: python | cuda | cpp
PROJECTS: list[dict] = [
    {"id": "001", "title": "亮度 / 灰度", "level": 1, "category": "post", "stack": "Python + Numba CUDA", "kind": "python", "entry": "brightness_grayscale.py"},
    {"id": "002", "title": "高斯模糊 shared mem", "level": 1, "category": "post", "stack": "Python + Numba CUDA", "kind": "python", "entry": "gaussian_blur.py"},
    {"id": "003", "title": "Sobel 边缘", "level": 1, "category": "post", "stack": "Python + Numba CUDA", "kind": "python", "entry": "sobel_edge.py"},
    {"id": "004", "title": "Tone Mapping", "level": 2, "category": "post", "stack": "Python + Numba CUDA", "kind": "python", "entry": "tone_mapping.py"},
    {"id": "005", "title": "HE + CLAHE", "level": 2, "category": "post", "stack": "Python + Numba CUDA", "kind": "python", "entry": "hist_clahe.py"},
    {"id": "006", "title": "多 Pass 管线", "level": 2, "category": "post", "stack": "Python + Numba CUDA", "kind": "python", "entry": "pipeline.py"},
    {"id": "007", "title": "Bilateral Filter", "level": 2, "category": "post", "stack": "C++ CUDA", "kind": "cuda", "entry": "bilateral"},
    {"id": "008", "title": "FFT 频域滤波", "level": 3, "category": "post", "stack": "C++ cuFFT", "kind": "cuda", "entry": "fft_filter"},
    {"id": "009", "title": "最简单光追", "level": 2, "category": "rt", "stack": "C++ CUDA", "kind": "cuda", "entry": "simple_rt"},
    {"id": "010", "title": "多物体 + 阴影 + 反射", "level": 2, "category": "rt", "stack": "C++ CUDA", "kind": "cuda", "entry": "scene_rt"},
    {"id": "011", "title": "三角网格 + BVH", "level": 3, "category": "rt", "stack": "C++ CUDA", "kind": "cuda", "entry": "mesh_bvh"},
    {"id": "012", "title": "AA + 景深", "level": 3, "category": "rt", "stack": "C++ CUDA", "kind": "cuda", "entry": "aa_dof"},
    {"id": "013", "title": "材质系统", "level": 3, "category": "rt", "stack": "C++ CUDA", "kind": "cuda", "entry": "materials"},
    {"id": "014", "title": "AO + GI 路径追踪", "level": 4, "category": "rt", "stack": "C++ CUDA", "kind": "cuda", "entry": "path_gi"},
    {"id": "015", "title": "光追优化", "level": 4, "category": "rt", "stack": "C++ CUDA", "kind": "cuda", "entry": "rt_opt"},
    {"id": "016", "title": "运动模糊 + 体积", "level": 4, "category": "rt", "stack": "C++ CUDA", "kind": "cuda", "entry": "volume_mb"},
    {"id": "017", "title": "小型 PBRT", "level": 5, "category": "rt", "stack": "C++ CUDA", "kind": "cuda", "entry": "mini_pbrt", "args": ["scenes/demo.json"]},
    {"id": "018", "title": "管线立方体 Phong", "level": 2, "category": "realtime", "stack": "C++ 软件光栅", "kind": "cpp", "entry": "soft_gl"},
    {"id": "019", "title": "纹理 + 法线", "level": 2, "category": "realtime", "stack": "C++ 软件光栅", "kind": "cpp", "entry": "texture_normal"},
    {"id": "020", "title": "Shadow Mapping", "level": 3, "category": "realtime", "stack": "C++ 软件光栅", "kind": "cpp", "entry": "shadow_map"},
    {"id": "021", "title": "PBR", "level": 3, "category": "realtime", "stack": "C++ CUDA", "kind": "cuda", "entry": "pbr"},
    {"id": "022", "title": "延迟渲染", "level": 3, "category": "realtime", "stack": "C++ 软件光栅", "kind": "cpp", "entry": "deferred"},
    {"id": "023", "title": "屏幕空间效果", "level": 3, "category": "realtime", "stack": "Python + Numba CUDA", "kind": "python", "entry": "ss_effects.py"},
    {"id": "024", "title": "地形 LOD", "level": 3, "category": "realtime", "stack": "C++ CUDA", "kind": "cuda", "entry": "terrain"},
    {"id": "025", "title": "迷你引擎", "level": 4, "category": "realtime", "stack": "C++ 软件光栅", "kind": "cpp", "entry": "mini_engine"},
    {"id": "026", "title": "粒子基础", "level": 2, "category": "particles", "stack": "Python + Numba CUDA", "kind": "python", "entry": "particles.py"},
    {"id": "027", "title": "烟花", "level": 2, "category": "particles", "stack": "Python + Numba CUDA", "kind": "python", "entry": "fireworks.py"},
    {"id": "028", "title": "SPH 流体", "level": 3, "category": "particles", "stack": "Python + Numba CUDA", "kind": "python", "entry": "sph.py"},
    {"id": "029", "title": "体积火焰", "level": 3, "category": "particles", "stack": "Python + Numba CUDA", "kind": "python", "entry": "volume_fire.py"},
    {"id": "030", "title": "布料", "level": 3, "category": "particles", "stack": "Python + Numba CUDA", "kind": "python", "entry": "cloth.py"},
    {"id": "031", "title": "碎片化", "level": 3, "category": "particles", "stack": "Python + Numba CUDA", "kind": "python", "entry": "fracture.py"},
    {"id": "032", "title": "视频滤镜管线", "level": 3, "category": "video", "stack": "Python + Numba CUDA", "kind": "python", "entry": "video_pipeline.py"},
    {"id": "033", "title": "光流", "level": 3, "category": "video", "stack": "Python + Numba CUDA", "kind": "python", "entry": "optical_flow.py"},
    {"id": "034", "title": "风格迁移", "level": 3, "category": "video", "stack": "Python + Numba CUDA", "kind": "python", "entry": "style_transfer.py"},
    {"id": "035", "title": "超分辨率", "level": 3, "category": "video", "stack": "Python + Numba CUDA", "kind": "python", "entry": "super_res.py"},
    {"id": "036", "title": "全景拼接", "level": 3, "category": "video", "stack": "Python + Numba CUDA", "kind": "python", "entry": "panorama.py"},
    {"id": "037", "title": "体渲染", "level": 3, "category": "sci", "stack": "Python + Numba CUDA", "kind": "python", "entry": "volume_render.py"},
    {"id": "038", "title": "流场", "level": 3, "category": "sci", "stack": "Python + Numba CUDA", "kind": "python", "entry": "flow_viz.py"},
    {"id": "039", "title": "Marching Cubes", "level": 3, "category": "sci", "stack": "Python + Numba CUDA", "kind": "python", "entry": "marching_cubes.py"},
    {"id": "040", "title": "点云", "level": 3, "category": "sci", "stack": "Python + Numba CUDA", "kind": "python", "entry": "point_cloud.py"},
    {"id": "041", "title": "地震体", "level": 3, "category": "sci", "stack": "Python + Numba CUDA", "kind": "python", "entry": "seismic_viz.py"},
    {"id": "042", "title": "噪声纹理", "level": 2, "category": "proc", "stack": "Python + Numba CUDA", "kind": "python", "entry": "noise_tex.py"},
    {"id": "043", "title": "程序化地形", "level": 3, "category": "proc", "stack": "Python + Numba CUDA", "kind": "python", "entry": "proc_terrain.py"},
    {"id": "044", "title": "分形", "level": 3, "category": "proc", "stack": "Python + Numba CUDA", "kind": "python", "entry": "fractal.py"},
    {"id": "045", "title": "植被", "level": 3, "category": "proc", "stack": "Python + Numba CUDA", "kind": "python", "entry": "vegetation.py"},
    {"id": "046", "title": "稳定流体", "level": 4, "category": "phys", "stack": "Python + Numba CUDA", "kind": "python", "entry": "stable_fluids.py"},
    {"id": "047", "title": "破碎仿真", "level": 4, "category": "phys", "stack": "Python + Numba CUDA", "kind": "python", "entry": "fracture_sim.py"},
    {"id": "048", "title": "FFT 海洋", "level": 4, "category": "phys", "stack": "Python + Numba CUDA", "kind": "python", "entry": "ocean.py"},
    {"id": "049", "title": "神经 / 引导降噪", "level": 4, "category": "ai", "stack": "Python + Numba CUDA", "kind": "python", "entry": "denoise.py"},
    {"id": "050", "title": "NeRF", "level": 5, "category": "ai", "stack": "Python + Numba CUDA", "kind": "python", "entry": "nerf.py"},
]

BY_ID = {p["id"]: p for p in PROJECTS}


def category_label(cid: str) -> str:
    for c in CATEGORIES:
        if c["id"] == cid:
            return c["label"]
    return cid
