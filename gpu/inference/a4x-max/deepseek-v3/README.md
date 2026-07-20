# DeepSeek-R1 (V3 系) · GB300 NVL72 · SGLang NVFP4 推理复现

> DeepSeek-R1-0528（V3 架构，671B 全注意力 MLA + MoE）在 **GB300 NVL72 (A4X Max / GKE)** 上用 SGLang 做 **NVFP4 + PD 分离 + Wide-EP** 长上下文推理的完整复现记录。
> 复现对象：LMSYS 官方博客 [*Deploying DeepSeek on GB300 NVL72*](https://lmsys.org/blog/2026-02-19-gb300-longctx/)（2026-02-19）+ [sglang#18703 recipe](https://github.com/sgl-project/sglang/issues/18703)。

## 文档导航

| 文档 | 用途 | 何时看 |
|---|---|---|
| [**sglang-r1-nvfp4-gb300-3p2d-DEPLOY-GUIDE.md**](./sglang-r1-nvfp4-gb300-3p2d-DEPLOY-GUIDE.md) | **端到端复现指南**（一次性、可照抄）：建 pod → bootstrap → 3P2D 启动 → router → benchmark。全程 **Local SSD RAID** 加载权重。 | **想跑通 R1 从这里开始** |
| [sglang-r1-nvfp4-128k-gb300.md](./sglang-r1-nvfp4-128k-gb300.md) | 复现方案 / 背景：官方博客解读、架构原理、参数由来。 | 想理解「为什么这么配」 |
| [sglang-r1-nvfp4-128k-gb300-RUNLOG.md](./sglang-r1-nvfp4-128k-gb300-RUNLOG.md) | 实测流水账：每轮（1→64 GPU）的命令、结果、踩坑、修法、benchmark。含坑速查。 | 遇到报错 / 想看原始数据 |

## 关键结论

- **部署形态**：3 prefill + 2 decode（DEP8 Wide-EP），KV cache 走**域内 NVLink**（mooncake `MC_FORCE_MNNVL=1`）。
- **规模递进**：单节点 → 1P1D → 3P2D（20 GPU）→ 64 GPU 满域 128K 长上下文 → +MTP（EAGLE spec decode，per-user 2.15×）。
- **benchmark**（8K/1K，warm，total in+out ÷ GPU）：64 GPU 峰值 **1,359 tok/s/GPU**；128K 长上下文对标官方 226 TPS/GPU。
- **存储**：权重全程从 GCS → **Local SSD RAID `/mnt/disks/raid/0`** 加载（不用内存盘，省 RAM 给 KV cache）。RAID 搭建见 [../deepseek-v4/gb300-local-ssd-raid0-SETUP.md](../deepseek-v4/gb300-local-ssd-raid0-SETUP.md)。

## 环境

- 集群：`gb300-gke-test`（GKE, us-central1, `tencent-gcp-taiji-poc`），机型 `a4x-maxgpu-4g-metal`（4 GPU/节点，277GB HBM 各）。
- 镜像：`lmsysorg/sglang:v0.5.15.post1-cu130`（含 sm_103a）。
- 模型：`nvidia/DeepSeek-R1-0528-NVFP4`（385G），GCS 备份 `gs://chrisya-gb300-models/DeepSeek-R1-0528-NVFP4-v2/`。

> V4（Flash / Pro，CSA+HCA 百万上下文）见 [../deepseek-v4/](../deepseek-v4/)。
