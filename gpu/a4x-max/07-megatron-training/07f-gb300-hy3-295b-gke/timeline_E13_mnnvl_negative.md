日志 1131 行，全程 504.9s

| 阶段 | 到达时刻 | 本阶段耗时 | 占启动 | 说明 |
|---|---|---|---|---|
| 进程启动 |      0.3s |     0.0s |      — | torchrun 拉起 python，第一行输出 |
| Python import |     30.5s |    30.1s |   8.6% | torch/TE/vLLM/modelopt 等重型包导入 |
| HF config 拉取 |     31.9s |     1.4s |   0.4% | qwen3 骨架 recipe 取 config |
| NCCL 初始化 |     69.5s |    37.6s |  10.7% | torch.distributed init + NCCL bootstrap |
| 模型构建 |     67.6s |    -1.9s |  -0.5% | GPTModel 实例化 + 权重分配 |
| 优化器构建 |     67.9s |     0.3s |   0.1% | distributed optimizer + 主权重分配 |
| DDP/梯度buffer |     67.7s |    -0.2s |  -0.0% | param_and_grad_buffer 分配 |
| setup 完成 |     70.2s |     2.5s |   0.7% | 数据加载器 + rerun state 就绪 |
| 进训练循环 |     70.8s |     0.6s |   0.2% | warmup 步开始 |
| CUDA graph capture 开始 |    406.5s |   335.7s |  95.9% | full_iteration 图捕获 |
| CUDA graph capture 结束 |    433.1s |    26.6s |   7.6% | 捕获完成 |
| 首个稳态步 |    350.4s |   -82.7s | -23.6% | 第一条吞吐记录 |

**启动总耗时（进程启动 → 首个稳态步）：350.1s**
