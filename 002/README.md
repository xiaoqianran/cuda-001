# 002 · GPU 高斯模糊

| 项目 | 内容 |
|------|------|
| 难度 | Lv.1 |
| 语言 | Python + Numba CUDA |
| 内容 | 5×5 / 15×15 高斯核 GPU 卷积 |
| 收获 | tile + halo + 共享内存并行卷积 |

## 运行

```bash
python gaussian_blur.py
```

## 要求覆盖
- `@cuda.jit` + **shared memory** 缓存 tile
- 边界：clamp / reflect / wrap
- 对比不同核大小耗时
