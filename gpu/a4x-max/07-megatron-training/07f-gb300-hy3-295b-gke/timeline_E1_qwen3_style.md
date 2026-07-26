日志 1129 行，全程 453.5s

| 阶段 | 到达时刻 | 本阶段耗时 | 占启动 | 说明 |
|---|---|---|---|---|
| 进程启动 |      0.3s |     0.0s |      — | torchrun 拉起 python，第一行输出 |
| Python import |     28.0s |    27.6s |   9.2% | torch/TE/vLLM/modelopt 等重型包导入 |
| HF config 拉取 |     29.2s |     1.2s |   0.4% | qwen3 骨架 recipe 取 config |
| NCCL 初始化 |     45.9s |    16.7s |   5.6% | torch.distributed init + NCCL bootstrap |
| 模型构建 |     44.2s |    -1.7s |  -0.6% | GPTModel 实例化 + 权重分配 |
| 优化器构建 |     44.5s |     0.4s |   0.1% | distributed optimizer + 主权重分配 |
| DDP/梯度buffer |     44.4s |    -0.2s |  -0.1% | param_and_grad_buffer 分配 |
| setup 完成 |     46.1s |     1.8s |   0.6% | 数据加载器 + rerun state 就绪 |
| 进训练循环 |     46.7s |     0.6s |   0.2% | warmup 步开始 |
| CUDA graph capture 开始 |    356.1s |   309.5s | 103.1% | full_iteration 图捕获 |
| CUDA graph capture 结束 |    381.6s |    25.4s |   8.5% | 捕获完成 |
| 首个稳态步 |    300.6s |   -81.0s | -27.0% | 第一条吞吐记录 |

**启动总耗时（进程启动 → 首个稳态步）：300.2s**
