# RUNBOOK — 在 G4 上复现 GLM-5.3-Flash 的最优配置

按顺序一步不差地执行即可复现 README 和 HTML 报告里的全部数字。每一步都写了「做完应该看到什么」。

尖括号 `<...>` 换成你自己的值：

| 占位符 | 含义 |
|---|---|
| `<your-project>` | GCP 项目 ID |
| `<your-bucket>` | 你自己的 GCS 桶（可选，缓存权重用，要和机器同区域） |
| `<A_IP>` `<B_IP>` `<C_IP>` | 各台机器的内网 IP（`hostname -I \| awk '{print $1}'`） |

**要几台机器**：配置 A（不分离）1 台；配置 B（2P2D）2 台；配置 C（4P2D）3 台。

| 步骤 | 耗时（实测） |
|---|---|
| 建一台 Spot 机器 | 1–2 分钟 |
| 装 Docker + 拉镜像 | 1–2 分钟 |
| 下权重：HuggingFace / 同区域 GCS 桶 | FP8 306 GiB 约 2.5 分钟 / 49 秒；NVFP4 185 GiB 约 1.5 分钟（HF） |
| 起服务（单实例到可用） | 约 6.5 分钟 |
| 一组并发扫描（1 → 256） | 约 35 分钟 |
| 一组速率扫描（5–7 档） | 约 20–25 分钟 |

---

## 0. 准备（在你自己的电脑上）

```bash
gcloud auth login
gcloud config set project <your-project>
gcloud services enable compute.googleapis.com

git clone https://github.com/yangwhale/gpu-tpu-pedia.git
cd gpu-tpu-pedia/gpu/inference/g4/glm-5.3-flash

export PROJECT=<your-project>
export ZONE=us-central1-b          # 需是有 G4 的 zone
```

配额：每台 `g4-standard-384` 需要 8 张 `nvidia-rtx-pro-6000`（Spot 用 preemptible 配额）。模型公开，不需要 HuggingFace token。

脚本目录：

| 路径 | 用途 |
|---|---|
| `scripts/create-vm.sh` | 建一台 G4 Spot 机器 |
| `scripts/setup-node.sh` | 在机器上装 Docker、拉镜像、拷权重 |
| `scripts/download-hf.sh` | 从 HuggingFace 下载权重 |
| `scripts/serve/instance.sh` | 起一个不分离的 vLLM 实例（可指定 GPU 子集） |
| `scripts/serve/pd.sh` | 起一个 PD 实例（P 或 D），已带 UCX 调优 |
| `scripts/serve/proxy.sh` + `toy_proxy_server.py` | PD 代理（vLLM 官方 NIXL 示例代理） |
| `scripts/bench/concurrency.sh` | 并发扫描（8K/1K，1 → 256） |
| `scripts/bench/rate.sh` | 泊松速率扫描 |
| `scripts/bench/run-pd.sh` | PD 一键：等实例 → 起代理 → 冒烟 → 并发扫描 |
| `scripts/bench/capacity.sh` | 单实例 prefill / decode 容量（算 P:D 配比） |
| `scripts/eval/gsm8k.sh` | GSM8K 能力评测 |
| `scripts/net/nixl_bench.py` | 独立 NIXL 传输基准 |
| `scripts/report/goodput.py`、`build_report.py` | 按延迟门槛算每卡吞吐、生成 HTML 报告 |
| `scripts/run-all.sh` | FP8 基线一键（装环境 → 下权重 → 起 TP8 → 冒烟 → 基础压测） |

## 1. 建机器（每台都一样）

```bash
NAME=g4-a PROJECT=$PROJECT ZONE=$ZONE bash scripts/create-vm.sh
# 需要几台就建几台：g4-b、g4-c
```

`create-vm.sh` 里的命令（可以直接复制）：

```bash
gcloud compute instances create $NAME --project $PROJECT --zone $ZONE \
  --machine-type g4-standard-384 \
  --provisioning-model SPOT --instance-termination-action STOP \
  --maintenance-policy TERMINATE \
  --image-family common-cu129-ubuntu-2404-nvidia-580 --image-project deeplearning-platform-release \
  --boot-disk-size 2000GB --boot-disk-type hyperdisk-balanced \
  --scopes cloud-platform --metadata install-nvidia-driver=True
```

| 参数 | 为什么 |
|---|---|
| `g4-standard-384` | 8 × 96 GB；FP8 306 GiB / NVFP4 185 GiB 都放得下 |
| `SPOT` + `--instance-termination-action STOP` | 测试省钱；被抢占时停机不删盘，`start` 回来权重还在。生产去掉这两项 |
| DLVM `common-cu129-ubuntu-2404-nvidia-580` | 预装驱动（实测 580.178.04） |
| 2 TB Hyperdisk Balanced | 两套权重 + 镜像（解压 35.5 GB）有余量 |
| `--scopes cloud-platform` | 机器上能直接 `gcloud storage` 读写你的桶 |

多台机器要在**同一 zone、同一 VPC**（默认 VPC 的 `default-allow-internal` 即可互通）。

**验证**：`gcloud compute ssh $NAME --zone $ZONE --project $PROJECT --command "nvidia-smi -L"` 列出 8 张 `RTX PRO 6000 Blackwell Server Edition`。首次 ssh 若报 `Permission denied (publickey)`，等十几秒重试。

把脚本拷上去：

```bash
gcloud compute scp --recurse scripts $NAME:~/ --zone $ZONE --project $PROJECT
```

## 2. 每台机器准备环境

在每台机器上：

```bash
# FP8 权重已在同区域桶里就设 MODEL_URI（49 秒拷完）；没有桶就不设 MODEL_URI，改用下面的 download-hf.sh
MODEL_URI=gs://<your-bucket>/models/zai-org/GLM-5.3-Flash bash ~/scripts/setup-node.sh

# NVFP4 权重（推荐配置都用它）
REPO=RedHatAI/GLM-5.3-Flash-NVFP4 DEST=/data/GLM-5.3-Flash-NVFP4 bash ~/scripts/download-hf.sh
# 或从桶：gcloud storage cp -r gs://<your-bucket>/models/RedHatAI/GLM-5.3-Flash-NVFP4 /data/
# FP8 没走桶时：REPO=zai-org/GLM-5.3-Flash DEST=/data/GLM-5.3-Flash bash ~/scripts/download-hf.sh
```

`setup-node.sh` 做三件事：装 Docker 与 NVIDIA Container Toolkit、拉固定 digest 的 vLLM 镜像、拷 FP8 权重。

**验证**：
- `sudo docker run --rm --gpus all ubuntu nvidia-smi -L` 看到 8 张卡；
- `du -sh /data/GLM-5.3-Flash /data/GLM-5.3-Flash-NVFP4` 约 306 GiB / 185 GiB。

> DLVM 镜像预装并 hold 了 `nvidia-container-toolkit 1.19.1`，安装时报 `held broken packages` **不影响使用**，只要上面的验证通过。系统 python 没带 pip，下载脚本用 uv 建独立环境。

**第一次下完权重，建议传一份到同区域桶**，以后拷贝只要 1 分钟以内：

```bash
gcloud storage rsync -r /data/GLM-5.3-Flash-NVFP4 gs://<your-bucket>/models/RedHatAI/GLM-5.3-Flash-NVFP4 --exclude="^\.cache/"
```

## 3. 配置 A：NVFP4 不分离（每台机器 2 × TP4）

每台机器上：

```bash
NV="MODEL_DIR=/data/GLM-5.3-Flash-NVFP4 EXTRA_ARGS=--no-enable-flashinfer-autotune"
env $NV NAME=glm-a GPUS=0,1,2,3 PORT=8001 bash ~/scripts/serve/instance.sh
env $NV NAME=glm-b GPUS=4,5,6,7 PORT=8002 bash ~/scripts/serve/instance.sh
```

`instance.sh` 的关键参数：`--tensor-parallel-size 4`（由 `GPUS` 推出）、`--max-model-len 131072`、`--max-num-seqs 256`、`--gpu-memory-utilization 0.85`、`--kv-cache-dtype fp8`、`--tool-call-parser glm47`、`--reasoning-parser glm45`。

**验证**（约 6.5 分钟后）：`curl -s localhost:8001/v1/models`、`curl -s localhost:8002/v1/models` 都返回 `glm`；日志有 `Using FLASHINFER_MLA_SPARSE_SM120 attention backend`、`Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend`。

客户端前面放任意 HTTP 负载均衡（按连接数或轮询），把请求分到所有实例的 `:8001` / `:8002`。

## 4. 配置 B：NVFP4 2P2D（两台机器，全 TP4）

| 机器 | 实例 |
|---|---|
| A | 2 个 P（GPU 0–3 → :8100，GPU 4–7 → :8101），另跑代理 :8000 |
| B | 2 个 D（GPU 0–3 → :8200，GPU 4–7 → :8201） |

```bash
NV="MODEL_DIR=/data/GLM-5.3-Flash-NVFP4 EXTRA_ARGS=--no-enable-flashinfer-autotune"
# 机器 A（P）
env $NV ROLE=prefill NAME=glm-p1 GPUS=0,1,2,3 PORT=8100 SIDE_PORT=5600 MAX_NUM_SEQS=64 GPU_UTIL=0.92 bash ~/scripts/serve/pd.sh
env $NV ROLE=prefill NAME=glm-p2 GPUS=4,5,6,7 PORT=8101 SIDE_PORT=5700 MAX_NUM_SEQS=64 GPU_UTIL=0.92 bash ~/scripts/serve/pd.sh
# 机器 B（D）
env $NV ROLE=decode NAME=glm-d1 GPUS=0,1,2,3 PORT=8200 SIDE_PORT=5600 MAX_NUM_SEQS=128 GPU_UTIL=0.92 bash ~/scripts/serve/pd.sh
env $NV ROLE=decode NAME=glm-d2 GPUS=4,5,6,7 PORT=8201 SIDE_PORT=5700 MAX_NUM_SEQS=128 GPU_UTIL=0.92 bash ~/scripts/serve/pd.sh
```

`pd.sh` 已设好的关键项（不要删）：

| 项 | 值 | 为什么 |
|---|---|---|
| `--kv-transfer-config` | `NixlConnector`，P 为 `kv_producer`、D 为 `kv_consumer` | KV 由 D 从 P 读取 |
| `VLLM_SSM_CONV_STATE_LAYOUT` | `DS` | 模型有 KDA 线性注意力层，NIXL 传状态要求 DS 布局；不设会在初始化末尾退出 |
| `UCX_TCP_TX_SEG_SIZE` / `UCX_TCP_RX_SEG_SIZE` | `1m` | 默认分段下 GPU 显存经 TCP 每连接约 5 Gbps，KV 传输慢 3 倍；1m 后约 27 Gbps/连接 |
| `VLLM_NIXL_SIDE_CHANNEL_HOST` / `_PORT` | 本机内网 IP / `SIDE_PORT` | 跨机握手；同机两个实例端口错开（5600 / 5700） |
| `MAX_NUM_SEQS=64 GPU_UTIL=0.92`（P） | | P 只做 prefill；TP4 的 PD 实例用默认值会报 `No available memory for the cache blocks` |
| `MAX_NUM_SEQS=128 GPU_UTIL=0.92`（D） | | 同上 |

**P 和 D 必须用相同的 TP**：P=TP4、D=TP8 时 NIXL 握手失败，所有请求 500。

代理 + 冒烟 + 并发扫描（在机器 A 上）：

```bash
TAG=2p2d-nvfp4 TOKENIZER=/data/GLM-5.3-Flash-NVFP4 CONCURRENCY='1 2 4 8 16 32 64 128 256' \
  P='<A_IP>:8100 <A_IP>:8101' D='<B_IP>:8200 <B_IP>:8201' bash ~/scripts/bench/run-pd.sh
# 结果：~/g4glm/pd-2p2d-nvfp4/，看到 "=== STEP ALL_DONE" 即完成
```

或分步：`PREFILL="<A_IP>:8100 <A_IP>:8101" DECODE="<B_IP>:8200 <B_IP>:8201" bash ~/scripts/serve/proxy.sh`，之后客户端只连代理的 `:8000`，接口与单机完全相同。

**验证 KV 传输已调优**：`sudo docker logs glm-d1 | grep "KV Transfer metrics"`，`Avg xfer time (ms)` 应在 120 ms 左右（未调优时 350 ms 以上）。

## 5. 配置 C：NVFP4 4P2D（三台机器，全 TP4）

机器 A、B 各按第 4 节起 2 个 P（:8100 / :8101），机器 C 按第 4 节起 2 个 D（:8200 / :8201），然后在 A 上：

```bash
TAG=4p2d-nvfp4 TOKENIZER=/data/GLM-5.3-Flash-NVFP4 CONCURRENCY='1 2 4 8 16 32 64 128 256' \
  P='<A_IP>:8100 <A_IP>:8101 <B_IP>:8100 <B_IP>:8101' D='<C_IP>:8200 <C_IP>:8201' bash ~/scripts/bench/run-pd.sh
```

## 6. 基线：FP8 单机 TP8

```bash
NAME=glm PORT=8000 bash ~/scripts/serve/instance.sh        # 默认 MODEL_DIR=/data/GLM-5.3-Flash，8 卡 TP8
```

或一键（装环境 → 下权重 → 起服务 → 冒烟 → 基础压测）：`setsid nohup bash ~/scripts/run-all.sh > ~/g4glm-run.log 2>&1 &`。

**验证**：日志里 `Model loading took 39.38 GiB memory`（每卡）、`GPU KV cache size: 4,441,368 tokens`。

## 7. 冒烟测试

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm", "messages": [{"role": "user", "content": "用一句话介绍香港。"}],
  "max_tokens": 2000, "reasoning_effort": "low"}'
```

应直接返回中文答案。这个模型默认以最高档（`max`）思考：`max_tokens` 给少了会全部花在思考上，`content` 为空、`finish_reason: length`——**不是服务坏了**。chat template **没有** `enable_thinking` 开关，控制思考用 `reasoning_effort`（`low` / `high`，不传为 `max`）。

## 8. 压测

压测工具是镜像自带的 `vllm bench serve`，**不用另装**；脚本用同一个镜像起一次性容器当客户端，需要一份模型目录给 tokenizer。

### 8.1 并发扫描（吞吐上限）

```bash
BASE=http://127.0.0.1:8000 TAG=<名字> TOKENIZER=/data/GLM-5.3-Flash-NVFP4 \
  CONCURRENCY='1 2 4 8 16 32 64 128 256' bash ~/scripts/bench/concurrency.sh
```

每档请求数 = 并发 × 4（至少 8），`--ignore-eos`，`--seed` 等于并发数。**不分离多实例**时，每个实例配一个客户端同时跑（`BASE=http://127.0.0.1:8001`、`:8002`，不同 `OUT`），结果相加。

### 8.2 速率扫描（真实延迟；性价比以它为准）

```bash
BASE=http://127.0.0.1:8000 TAG=<名字> TOKENIZER=/data/GLM-5.3-Flash-NVFP4 RATES='0.5 1 1.5 2 2.5' bash ~/scripts/bench/rate.sh
```

泊松到达，每档请求数 = max(60, 速率 × 180)，**每档不同随机种子**（同一种子会重发相同 prompt、命中前缀缓存，TTFT 偏低）。多实例时每个客户端发总速率的一份，并设不同的 `SEED_OFFSET`。

本报告用的速率：

| 部署 | RATES |
|---|---|
| 单机不分离（FP8 TP8 / NVFP4 TP8） | `0.25 0.5 0.75 1 1.25 (1.5)` |
| 单机 NVFP4 2×TP4（每个实例的速率） | `0.25 0.5 0.625 0.75 0.875 1.0` |
| 2P2D | `0.5 1 1.5 2 2.5` |
| 4P2D | `0.5 1 1.5 2 2.5 3 3.5` |

### 8.3 容量测试（算 P:D 配比）

对一个不分离实例：

```bash
BASE=http://127.0.0.1:8000 TAG=nvfp4-tp4 TOKENIZER=/data/GLM-5.3-Flash-NVFP4 bash ~/scripts/bench/capacity.sh
```

prefill：8K 输入、1 token 输出；decode：8,184 token 共享前缀（命中缓存）+ 1,024 token 输出。配比 = (D 每秒能完成的请求数 × 输入长度) ÷ P 的 prefill 吞吐。

## 9. 能力评测（GSM8K）

```bash
TAG=nvfp4 TOKENIZER=/data/GLM-5.3-Flash-NVFP4 bash ~/scripts/eval/gsm8k.sh   # 对 :8000（单实例或代理）
TAG=fp8   TOKENIZER=/data/GLM-5.3-Flash        bash ~/scripts/eval/gsm8k.sh
```

lm-evaluation-harness `0.4.13` + `transformers` 自动装进独立环境；全量 1,319 题、5-shot、`temperature=0`、并发 64，约 4 分钟。结果在 `~/g4glm/eval-<TAG>/`。

## 10. 网络基准（可选，排查 PD 传输用）

```bash
IMG=vllm/vllm-openai@sha256:88cc2118dab845fec1ff759c2fc8afa517cf8a668162ed8683b2f1b781372f57
UCX="-e UCX_TCP_TX_SEG_SIZE=1m -e UCX_TCP_RX_SEG_SIZE=1m"
# 机器 1（被读）
sudo docker run --rm --gpus '"device=0"' --network host $UCX -v ~/scripts/net:/n --entrypoint python3 $IMG /n/nixl_bench.py target --mem cuda --total-mb 80
# 机器 2（读取）
sudo docker run --rm --gpus '"device=0"' --network host $UCX -v ~/scripts/net:/n --entrypoint python3 $IMG /n/nixl_bench.py init --host <机器1_IP> --mem cuda --total-mb 80 --descs 300 --iters 10
```

期望约 25 ms / 27 Gbps；去掉 `$UCX` 约 128 ms / 5 Gbps。多对并行：每张卡一对（`--gpus device=i`、`--port 18515+i`），8 对可打满单网卡（约 176 Gbps），用 `/proc/net/dev` 的 ens3 字节数核对。机器间带宽基线：`iperf3 -c <IP> -P 16 -t 10`（实测 173 Gbit/s）。

## 11. 性价比与报告

```bash
python3 scripts/report/goodput.py        # 按延迟门槛算每卡吞吐（门槛在文件顶部 SLAS 里改）
python3 scripts/report/build_report.py   # 从 results/ 重新生成 report.html
```

## 12. 对比测试：H200（A3 Ultra）

同一套脚本，机型换成 `a3-ultragpu-8g`（其余 `create-vm.sh` 参数不变；A3 Ultra 的 Spot 容量紧张，可在多个 zone 轮流重试），只测 FP8：

```bash
NAME=glm PORT=8000 bash ~/scripts/serve/instance.sh                                        # TP8
NAME=glm-a GPUS=0,1,2,3 PORT=8001 bash ~/scripts/serve/instance.sh                          # 或 2×TP4
NAME=glm-b GPUS=4,5,6,7 PORT=8002 bash ~/scripts/serve/instance.sh
```

压测同第 8 节（本报告 H200 TP8 速率用 `0.25 0.5 0.75 1 1.25 1.5 2 2.5 3`，2×TP4 每实例 `0.25 0.5 0.75 1 1.25 1.5`）。

## 13. 收尾

```bash
gcloud compute instances delete g4-a g4-b g4-c --zone $ZONE --project $PROJECT
```

## 排障

| 现象 | 处理 |
|---|---|
| `content` 为空、`finish_reason: length` | 默认 `max` 思考档把 token 用完；调大 `max_tokens` 或传 `"reasoning_effort": "low"` |
| PD 启动约 6 分钟后退出：`3-read Mamba conv transfer requires DS conv state layout` | 加 `VLLM_SSM_CONV_STATE_LAYOUT=DS`（`pd.sh` 已带） |
| TP4 的 PD 实例报 `No available memory for the cache blocks` | P 用 `MAX_NUM_SEQS=64 GPU_UTIL=0.92`，D 用 `MAX_NUM_SEQS=128 GPU_UTIL=0.92` |
| PD 请求全部 500，D 日志 `handshake_failed`，前有 `Not pairing remote 2304-token blocks with local 1280-token blocks` | P 和 D 的 TP 不同；改成相同 TP |
| PD 的 KV 传输 300 ms 以上、首字慢 | 没设 `UCX_TCP_TX_SEG_SIZE=1m` / `UCX_TCP_RX_SEG_SIZE=1m` |
| 启动日志大量 `CUDACachingAllocator ... OOM` | 显存探测阶段的正常警告，不是失败原因；看 `RuntimeError` / `AssertionError` 那行 |
| 预热阶段 GPU 100% 卡住 | 加 `--no-enable-flashinfer-autotune`（NVFP4 已带） |
| 预热时刷 `No available shared memory broadcast block found in 60 seconds` | JIT 编译慢，不是挂了，等着 |
| 容量测试用 chat 接口发 1 token 请求报 `Never received a valid chunk` | 1 个 token 落进 reasoning 字段；`capacity.sh` 已改用 `/v1/completions` |
| Spot 被抢占 | `gcloud compute instances start <NAME>`，权重在系统盘不丢 |
