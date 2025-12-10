# gpu-tpu-pedia

GPU 和 TPU 学习与实践知识库

---

## 项目结构

### TPU

#### 1. HunyuanVideo-1.5 TPU 推理 ⭐

在 Google Cloud TPU v6e-8 上运行 HunyuanVideo-1.5 视频生成模型。

**关键特性：**
- 支持 TPU v6e-8（8 chips, 256GB HBM）
- JAX + torchax 实现 TPU 原生推理
- Splash Attention 优化
- 张量并行权重分片 + DeepCache 加速

**性能数据（121帧 720p, 50步）：**

| 模式 | 每步时间 | 总时间 | 加速比 |
|------|----------|--------|--------|
| 标准 TP | 8.12s | 6.8 分钟 | 1.0x |
| **TP + fc2 Replicated** | **7.29s** | **6.1 分钟** | **1.11x** |
| TP + DeepCache | ~4s | ~3.5 分钟 | ~2x |

**文档：**
- [项目 README](tpu/HunyuanVideo-1.5/README.md)
- [TPU 推理（原生版本）](tpu/HunyuanVideo-1.5/generate_hunyuan_flax_staged/README.md)
- [TPU 推理（Diffusers 版本）](tpu/HunyuanVideo-1.5/generate_diffusers_flax_staged/README.md)
- [GPU→TPU 迁移指南](tpu/HunyuanVideo-1.5/generate_hunyuan_flax_staged/TORCHAX_MIGRATION_GUIDE.md)
- [GPU 推理参考](tpu/HunyuanVideo-1.5/generate_hunyuan_gpu_staged/README.md)

---

#### 2. CogVideoX TPU 加速

在 TPU 上运行 CogVideoX 视频生成模型，JAX/Flax 原生实现。

**关键特性：**
- Splash Attention TPU 优化
- Flax VAE 解码器（解决 OOM）
- FSDP/Tensor Parallel 模型分片
- BFloat16 全流程优化

**性能：**
- 第一次运行（含编译）：~45 秒
- 后续运行：~18 秒
- 加速比：2.44x

**文档：**
- [CogVideoX README](tpu/cogvideo/README.md)

---

#### 3. CogVideoX VAE PyTorch→JAX 迁移 📚

完整的 PyTorch 到 JAX/Flax 迁移方法论，基于 2,013 行 VAE 代码的实战经验。

**核心价值：**
- 1,150+ 行迁移圣经文档
- 17 个单元测试全部通过
- 数值精度 MAE < 0.6
- JIT 加速 112x

**关键教训：**
1. **数据格式陷阱**：Channel-First vs Channel-Last
2. **GroupNorm 必须在 channel-first 计算**才能匹配 PyTorch
3. **JIT 是性能关键**：不仅快 100x+，还能解决 OOM
4. **时序模型特殊性**：CausalConv 不能时间分片

**文档：**
- [项目 README](tpu/cogvideo/cogvae_migration/README.md)
- [PyTorch→JAX 迁移圣经](tpu/cogvideo/cogvae_migration/docs/PYTORCH_TO_JAX_MIGRATION_BIBLE_ZH.md)
- [范式转换指南](tpu/cogvideo/cogvae_migration/docs/PYTORCH_TO_JAX_PARADIGM_SHIFT_ZH.md)

---

#### 4. PyTorch→JAX 入门教程

从零开始学习如何在 TPU 上运行 HuggingFace 模型。

**教程内容：**
1. [在 JAX 中运行 HuggingFace 模型](tpu/torch_to_jax_jumpstart/01-run-huggingface-model-in-jax-zh.md)
2. [分布式运行 HuggingFace 模型](tpu/torch_to_jax_jumpstart/02-run-huggingface-model-distributed-zh.md)
3. [进阶：使用 torchax](tpu/torch_to_jax_jumpstart/03-run-huggingface-model-in-jax-zh.md)
4. [完整示例代码](tpu/torch_to_jax_jumpstart/04-run-hugging-face-model-in-jax-zh.md)

---

#### 5. TPU 图像处理示例

展示 GPU→TPU 图像处理迁移，包含 crop/resize/blur 等操作。

**关键发现：**
- TPU tracing 首次较慢（0.58秒），后续快 20x（0.028秒）
- float32 精度与 GPU 完全一致
- bfloat16 精度下降 218 倍，需谨慎使用

**文档：**
- [图像处理 README](tpu/cogvideo/image_processing/README.md)

---

### GPU

#### 1. DeepEP on GKE B200

在 Google Kubernetes Engine (GKE) 上部署 DeepSeek 的 DeepEP 框架。

**关键特性：**
- NVIDIA B200 GPU（8x/node）
- RDMA 网络配置
- 节点内/节点间测试

**技术栈：**
- DOCA OFED v3.0.0
- NVIDIA Driver 575
- CUDA Toolkit 12.9
- NVSHMEM 3.2.5
- PyTorch (CUDA 12.9)

**文档：**
- [DeepEP README](gpu/deepep/README.md)

---

#### 2. HunyuanVideo-1.5 GPU 推理

在 NVIDIA H100 8卡上运行 HunyuanVideo-1.5 视频生成。

**关键特性：**
- Flash Attention 2/3、SageAttention、Sparse Attention
- Sequence Parallelism 多卡并行
- DeepCache 加速（1.83x）

**Attention 性能对比：**

| 模式 | 加速比 | 质量 | 推荐场景 |
|------|--------|------|----------|
| Flash Attention 2 | 1.0x | ✅ 最优 | 生产环境 |
| **DeepCache** | **1.83x** | ✅ 良好 | **日常使用** |
| SageAttention | 1.6x | ⚠️ 有损 | 快速预览 |

**文档：**
- [GPU 推理 README](tpu/HunyuanVideo-1.5/generate_hunyuan_gpu_staged/README.md)

---

## 贡献

欢迎提交 Issues 和 Pull Requests！

## 许可

本项目采用开源许可。详见各子项目的许可声明。
