// 公共：向量与 PPM（POD 友好，可用于 __constant__ 显存）
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <cuda_runtime.h>

// 无用户定义构造函数，满足 CUDA constant 内存 POD 要求
struct Vec3 {
    float x, y, z;
    __host__ __device__ Vec3 operator+(Vec3 o) const { return Vec3{x + o.x, y + o.y, z + o.z}; }
    __host__ __device__ Vec3 operator-(Vec3 o) const { return Vec3{x - o.x, y - o.y, z - o.z}; }
    __host__ __device__ Vec3 operator*(Vec3 o) const { return Vec3{x * o.x, y * o.y, z * o.z}; }
    __host__ __device__ Vec3 operator*(float s) const { return Vec3{x * s, y * s, z * s}; }
    __host__ __device__ Vec3 operator/(float s) const { return Vec3{x / s, y / s, z / s}; }
    __host__ __device__ Vec3 operator-() const { return Vec3{-x, -y, -z}; }
};

__host__ __device__ inline Vec3 make_vec3(float a, float b, float c) {
    return Vec3{a, b, c};
}
__host__ __device__ inline Vec3 operator*(float s, Vec3 v) { return v * s; }
__host__ __device__ inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ inline Vec3 cross(Vec3 a, Vec3 b) {
    return Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
__host__ __device__ inline float len(Vec3 v) { return sqrtf(dot(v, v)); }
__host__ __device__ inline Vec3 normalize(Vec3 v) {
    float L = len(v);
    return L > 0 ? v / L : v;
}
__host__ __device__ inline Vec3 reflect(Vec3 v, Vec3 n) { return v - n * (2.f * dot(v, n)); }

struct Ray {
    Vec3 o, d;
};

inline void write_ppm(const std::string& path, const std::vector<uint8_t>& rgb, int w, int h) {
    FILE* f = fopen(path.c_str(), "wb");
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    fwrite(rgb.data(), 1, rgb.size(), f);
    fclose(f);
}
