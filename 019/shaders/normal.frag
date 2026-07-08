#version 330 core
// 法线贴图片元：TBN 变换到世界空间
in vec2 vUV; in mat3 vTBN; in vec3 vWorldPos;
uniform sampler2D uAlbedo, uSpecular, uNormal;
uniform vec3 uLightPos, uViewPos;
out vec4 FragColor;
void main(){
    vec3 albedo = texture(uAlbedo, vUV).rgb;
    float specMask = texture(uSpecular, vUV).r;
    vec3 nmap = texture(uNormal, vUV).xyz * 2.0 - 1.0;
    vec3 N = normalize(vTBN * nmap);
    vec3 L = normalize(uLightPos - vWorldPos);
    vec3 V = normalize(uViewPos - vWorldPos);
    vec3 R = reflect(-L, N);
    float diff = max(dot(N,L),0.0);
    float spec = pow(max(dot(R,V),0.0), 32.0) * specMask;
    FragColor = vec4(albedo*(0.15+0.7*diff) + vec3(spec), 1.0);
}
