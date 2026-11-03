# 测试过程记录

按时间记录每一步做了什么、为什么这么做、遇到什么。时间为 UTC（括号内为 HKT）。

## 2026-11-03

**02:35（10:35）调研结论**：vLLM 官方 recipe 不含 SM120；社区两条路线（vLLM FP8 TP8 / SGLang NVFP4 TP2）。选路线 A 先测，理由见 README 第 3 节。

**02:38（10:38）查预留**：项目里唯一的 G4 预留是 5 台 `g4-standard-48`（每台 1 卡），且 `specificReservationRequired: true`。
单卡装不下 306 GiB 权重，G4 跨机又没有 RDMA，几台单卡机拼不成 TP，所以这份预留用不上，改开 Spot `g4-standard-384`。

**02:40（10:40）建机**：us-central1-b 第一次就拿到 Spot 容量。8 卡、驱动 580.178.04、Ubuntu 24.04.5。

**02:41（10:41）镜像不存在**：原作者报告的 `vllm/vllm-openai:glm53-sm120nope` 在 Docker Hub 返回 404。同仓库存在的相关标签：

| 标签 | 最后更新 | 说明 |
|---|---|---|
| `nightly` | 2026-11-02 | 原生 SM120 后端，**本次采用** |
| `glm53-flash` / `glm53-flash-x86_64-cu130` | 2026-09-09 | 官方 GLM-5.3-Flash 镜像；社区报告在 SM120 上报 `pe_dim must be 64 for fp8_ds_mla` |

社区 overlay `cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1`（2026-08-26）留作备选。

**02:43（10:43）开跑**：`scripts/run-all.sh` 后台执行：装 Docker → 拉镜像与下载权重并行 → 起服务 → 冒烟 → 压测。

**02:51（10:51）第一次失败**：下载步骤报 `No module named pip` → `No module named 'huggingface_hub'`。DLVM 的 Ubuntu 24.04 镜像里系统 python3 不带 pip。
改用 uv 建独立虚拟环境装 `huggingface_hub`（RUNBOOK 第 4 步、`run-all.sh` 已同步修改）。镜像这时已拉完（`nightly`，解压后 35.5 GB）。

**02:53（10:53）重跑**：下载速度约 2 GB/s，开头 60 秒下了 115 GB。`run-all.sh` 同时加了 `env.txt` 采集（镜像 digest、vLLM 版本、驱动、拓扑）。
