# 工具和技能

## 推理服务器
- **sglang**: 推理服务器部署，有 sglang-installer skill
- **vllm**: 推理服务器部署，有 vllm-installer skill

## 分布式
- **Ray**: 分布式集群，有 ray-cluster-installer skill

## 终端环境
- **tmux**: oh-my-tmux 配置，有 tmux-installer skill
- **zsh**: oh-my-zsh + agnoster 主题，有 zsh-installer skill

## 多媒体生成
- **imagen-generator**: Imagen 4 文生图，同步 `predict` API，秒级返回
- **veo-generator**: Veo 3.1 文生视频，异步 `predictLongRunning` + polling，~60-90s

## 自定义 Skill
| Skill | 用途 |
|-------|------|
| imagen-generator | Imagen 4 文生图（Vertex AI） |
| veo-generator | Veo 3.1 文生视频（Vertex AI） |
| lssd-mounter | Local SSD RAID0 挂载到 /lssd |
| deepep-installer | DeepEP (DeepSeek Expert Parallelism) 安装 |
| parallel-ssh | 多主机并行 SSH 操作 |
| tpu-trainer | TPU v7 训练自动化 |
| feishu-report | 发送报告到飞书 |
| wechat-report | 发送报告到微信 |
| discord-report | 发送报告到 Discord |
| agent-teams | Agent Teams 并行任务团队管理 |
