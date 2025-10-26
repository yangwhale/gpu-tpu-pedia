# gpu-tpu-pedia

GPU 和 TPU 学习和实践知识库

## 项目结构

### GPU

#### DeepEP on GKE B200

在 Google Kubernetes Engine (GKE) 上部署和测试 DeepSeek 的 DeepEP (Deep Efficient Parallelism) 框架。

**关键特性：**
- 支持 NVIDIA B200 GPU
- 基于 Ubuntu 的 GKE 节点池
- RDMA 网络配置用于高速 GPU 间通信
- 完整的节点内和节点间测试支持

**快速开始：**
- [完整安装指南](gpu/deepep/README.md)
- [节点池创建配置](gpu/deepep/README.md#创建-gke-ubuntu-节点池)
- [DeepEP 安装程序](gpu/deepep/deepep-installer.yaml)
- [节点内测试](gpu/deepep/deepep-intranode.yaml)
- [节点间测试](gpu/deepep/deepep-internode.yaml)

**技术栈：**
- NVIDIA B200 GPU (8x per node)
- DOCA OFED v3.0.0
- NVIDIA Driver 575 (开源版本)
- CUDA Toolkit 12.9
- NVSHMEM 3.2.5 (sm_100 架构)
- gdrcopy v2.5.1
- PyTorch (CUDA 12.9)

### TPU

TPU 相关项目和实验（开发中）

## 贡献

欢迎提交 Issues 和 Pull Requests！

## 许可

本项目采用开源许可。详见各子项目的许可声明。
