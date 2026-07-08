#version 330 core
// Cook-Torrance PBR 片元（与 CUDA 核一致）
uniform vec3 uAlbedo; uniform float uMetallic, uRoughness;
uniform samplerCube uIrradiance; uniform samplerCube uPrefilter; uniform sampler2D uBRDFLUT;
// ... IBL 组合
