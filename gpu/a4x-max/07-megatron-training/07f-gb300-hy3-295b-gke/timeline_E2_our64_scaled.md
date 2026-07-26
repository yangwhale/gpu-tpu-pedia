日志 1759 行，全程 473.3s

| 阶段 | 到达时刻 | 本阶段耗时 | 占启动 | 说明 |
|---|---|---|---|---|
| 进程启动 |      0.3s |     0.0s |      — | torchrun 拉起 python，第一行输出 |
| Python import |     28.3s |    28.0s |  13.4% | torch/TE/vLLM/modelopt 等重型包导入 |
| HF config 拉取 |     29.6s |     1.3s |   0.6% | qwen3 骨架 recipe 取 config |
| NCCL 初始化 |     45.8s |    16.2s |   7.8% | torch.distributed init + NCCL bootstrap |
| 模型构建 |     44.4s |    -1.4s |  -0.7% | GPTModel 实例化 + 权重分配 |
| 优化器构建 |     44.9s |     0.5s |   0.2% | distributed optimizer + 主权重分配 |
| DDP/梯度buffer |     44.5s |    -0.4s |  -0.2% | param_and_grad_buffer 分配 |
| setup 完成 |     46.0s |     1.5s |   0.7% | 数据加载器 + rerun state 就绪 |
| 进训练循环 |     46.6s |     0.6s |   0.3% | warmup 步开始 |
| CUDA graph capture 开始 |    290.5s |   244.0s | 116.6% | full_iteration 图捕获 |
| 首个稳态步 |    209.5s |   -81.0s | -38.7% | 第一条吞吐记录 |

> 未命中的阶段标记（该配置下不存在或日志未打印）：CUDA graph capture 结束

**启动总耗时（进程启动 → 首个稳态步）：209.2s**
