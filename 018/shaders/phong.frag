// 片元着色器：Phong 着色
#version 330 core
in vec3 vNormal;
in vec3 vWorldPos;
uniform vec3 uLightPos;
uniform vec3 uViewPos;
uniform vec3 uAlbedo;
out vec4 FragColor;
void main(){
    vec3 N = normalize(vNormal);
    vec3 L = normalize(uLightPos - vWorldPos);
    vec3 V = normalize(uViewPos - vWorldPos);
    vec3 R = reflect(-L, N);
    float diff = max(dot(N,L), 0.0);
    float spec = pow(max(dot(R,V), 0.0), 32.0);
    vec3 color = uAlbedo * (0.15 + 0.7*diff) + vec3(1.0)*0.4*spec;
    FragColor = vec4(color, 1.0);
}
