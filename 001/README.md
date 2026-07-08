# 001 · GPU 亮度调节 & 灰度化

| 项目 | 内容 |
|------|------|
| 难度 | **Lv.1** |
| 语言 | Python + Numba CUDA |
| 内容 | 对一张 1080p 图像实现 GPU 版亮度调节和灰度化 |
| 收获 | 跑通 **CPU → GPU** 的第一个完整闭环 |

---

## 目标

1. 用 Numba `@cuda.jit` 编写 kernel  
2. 对比 NumPy CPU 版本的耗时  
3. 理解 **grid / block / thread** 的三维索引如何映射到二维像素  

---

## 环境

```bash
# 已验证环境：NVIDIA GPU + numba + numpy + pillow
python brightness_grayscale.py
```

依赖：`numba`、`numpy`、`pillow`（脚本会自动合成 1920×1080 测试图，无需外部图片）。

---

## 核心概念：索引映射

CUDA 的执行模型是三维的，但图像是二维的。本课用 **xy 平面**，`z` 恒为 1：

```
像素坐标 (x, y)：
  x = blockIdx.x * blockDim.x + threadIdx.x   # 列（宽）
  y = blockIdx.y * blockDim.y + threadIdx.y   # 行（高）
```

| 参数 | 本课取值 | 含义 |
|------|----------|------|
| `blockDim` | `(16, 16)` | 每个 block 有 16×16 = **256** 个线程 |
| `gridDim`  | `(120, 68)` | 盖满 1920×1080 需要的 block 数（向上取整） |
| 总线程数   | 1920×1088 | 略大于像素数 → **必须做边界检查** |

```python
if x >= width or y >= height:
    return   # 超出图像的线程什么都不做
```

示意图：

```
                block (0,0)          block (1,0)         ...
             ┌────────────────┐  ┌────────────────┐
             │ t(0,0) … t(15,0)│  │                │
  行 y ↑     │   …             │  │                │
             │ t(0,15)…t(15,15)│  │                │
             └────────────────┘  └────────────────┘
                  列 x →
```

---

## 两个 kernel

### 1. 亮度调节

```text
out[y, x, c] = clamp(src[y, x, c] * factor, 0, 255)
```

每个线程负责 **一个像素的 3 个通道**（R/G/B）。

### 2. 灰度化

```text
gray[y, x] = 0.299·R + 0.587·G + 0.114·B
```

每个线程负责 **一个像素 → 一个灰度值**。

---

## 完整数据流（CPU → GPU 闭环）

```
Host (NumPy)                Device (GPU)
─────────────               ─────────────
img (H,W,3) uint8
      │
      ├─ cuda.to_device ──►  d_src
      │                        │
      │                   kernel[grid, block]
      │                        │
      │                     d_dst
      │                        │
      ◄─ copy_to_host ─────────┘
result (NumPy)
      │
      ▼
与 CPU 结果逐像素比对 + 计时对比
```

注意：测 GPU 时间时要在计时区间末尾调用 `cuda.synchronize()`，否则只测到了「kernel 提交」而不是「算完」。

---

## 运行

```bash
cd 001
python brightness_grayscale.py
```

预期输出大致包括：

- GPU 型号、图像尺寸  
- grid / block 配置与索引公式  
- 亮度、灰度各自的 CPU / GPU 耗时与加速比  
- CPU 与 GPU 结果的最大像素差（亮度应为 0；灰度因浮点舍入可能 ≤1）  
- `output/` 下三张图：原图、提亮、灰度  

---

## 思考题（自测）

1. 为什么 `grid` 要用「向上取整」而不是直接 `width // 16`？  
2. 把 `block` 改成 `(32, 32)` 或 `(8, 8)`，耗时会怎样变？为什么？  
3. 如果图像是 4K（3840×2160），`grid` 应该怎么算？  
4. 本课把 RGB 三个通道放在同一个线程里处理。若改成「一个线程只算一个通道」，索引该怎么写？

---

## 文件

```
001/
├── README.md                 # 本说明
├── brightness_grayscale.py   # 完整可运行脚本
└── output/                   # 运行后生成
    ├── 00_original.png
    ├── 01_brightness_gpu.png
    └── 02_grayscale_gpu.png
```
