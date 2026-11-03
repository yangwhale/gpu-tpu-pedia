# 原始数据

压测跑完后放这里：

- `bench-in<输入>-out<输出>-c<并发>.txt` — `vllm bench serve` 每个场景的完整输出
- `server.log` — vLLM 服务启动与运行日志（含所选 attention 后端、KV 池大小、CUDA graph 捕获）
- `smoke.json` — 冒烟测试的原始返回
- `env.txt` — 镜像 digest、vLLM 版本、驱动、`nvidia-smi topo -m`
