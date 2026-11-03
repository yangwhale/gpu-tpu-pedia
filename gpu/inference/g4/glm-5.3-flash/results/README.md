# 原始数据

只放**最终配置**的结果。每个 `bench-*.txt` / `rate-*.txt` 是 `vllm bench serve` 的完整输出，同名 `.json` 是逐请求明细。负载统一为 8K 输入 / 1K 输出（基础场景除外）。

| 目录 | 内容 | 卡数 |
|---|---|---|
| `fp8-tp8-single/` | FP8 单机 TP8：基础 7 场景 + `8k1k/` 并发扫描；`server.log`、`env.txt`（镜像 digest、驱动、拓扑）、冒烟返回 | 8 |
| `rate-colocated-fp8-tp8/` | FP8 单机 TP8，不分离，泊松速率扫描 | 8 |
| `rate-colocated-nvfp4-tp8/` | NVFP4 单机 TP8，不分离，速率扫描 | 8 |
| `rate-colocated-nvfp4-2xtp4/` | NVFP4 单机 2×TP4，不分离，速率扫描（每实例一个客户端，`inst-*`） | 8 |
| `colocated-4xtp4-nvfp4/` | NVFP4 两机 4×TP4，不分离，并发扫描（`A/`、`B/` 两台机器各两个实例） | 16 |
| `pd-2p2d-nvfp4/`、`rate-2p2d-nvfp4/` | NVFP4 2P2D（全 TP4，UCX 分段 1m）：并发扫描、速率扫描、decode 日志（含 KV 传输统计） | 16 |
| `pd-4p2d-nvfp4/`、`rate-4p2d-nvfp4/` | NVFP4 4P2D（全 TP4，UCX 分段 1m） | 24 |
| `capacity/` | 单实例 prefill / decode 容量（算 P:D 配比用），NVFP4 TP8 与 TP4 | 8 / 4 |
| `eval/` | GSM8K：FP8 与 NVFP4 的 lm-eval 结果、逐题记录、配对分析 | — |
| `nixl-bench/` | 独立 NIXL 传输基准：UCX 参数、GPU/CPU 内存、多对并行 | — |

`_archive/` 是配置不对或已被替代的中间结果，**不用于结论**，保留只为可追溯，原因见其中的 README。
