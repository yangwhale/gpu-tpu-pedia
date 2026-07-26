日志 6 行，全程 1.0s

| 阶段 | 到达时刻 | 本阶段耗时 | 占启动 | 说明 |
|---|---|---|---|---|
| 进程启动 |      0.3s |     0.0s |      — | torchrun 拉起 python，第一行输出 |

> 未命中的阶段标记（该配置下不存在或日志未打印）：Python import, HF config 拉取, NCCL 初始化, 模型构建, 优化器构建, DDP/梯度buffer, setup 完成, 进训练循环, CUDA graph capture 开始, CUDA graph capture 结束, 首个稳态步

**启动总耗时（进程启动 → 首个稳态步）：0.0s**
