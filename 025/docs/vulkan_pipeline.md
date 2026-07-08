# Vulkan 管线对应说明（本项目用软件路径演示等价阶段）

1. Instance / Device / Queue
2. Swapchain（此处写 PPM 帧）
3. RenderPass + Framebuffer（GBuffer + 主输出）
4. Command Buffer 录制：Shadow → GBuffer → Lighting → Post
5. 同步：frame fence 概念 → 本实现顺序执行
6. ImGui 调试：stats.txt 输出 draw call / 时间戳
