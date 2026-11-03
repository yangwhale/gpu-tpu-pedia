# RUNBOOK — 在 G4 上跑 GLM-5.3-Flash（vLLM · FP8 · TP8）

从零开始，按顺序执行。每一步都写了「做完应该看到什么」。
一键版：第 1 步建好机器后，把 [`scripts/run-all.sh`](scripts/run-all.sh) 拷上去用 `setsid nohup` 跑，它按顺序做完第 2–7 步。
设了 `MODEL_URI=gs://...` 就从桶拷权重（4B），否则从 HuggingFace 下（4A）。

总耗时参考：建机 1–2 分钟；装 Docker 约 1 分钟；下载 306 GiB 权重：HuggingFace 约 2.5 分钟，同区域 GCS 桶 49 秒；服务启动约 6.5 分钟（权重加载 31 s + 编译/预热/CUDA graph 共 318 s）；基础压测 7 个场景约 8 分钟。

---

## 0. 前提

- 一个有 G4 配额的 GCP 项目（`g4-standard-384` 需要 8 张 `nvidia-rtx-pro-6000` 配额；Spot 用 Spot 配额）。
- 本地装好 `gcloud` 并已登录。
- 能访问 HuggingFace 和 Docker Hub（模型公开，不需要 HF token）。

```bash
export PROJECT=<your-project>
export ZONE=us-central1-b      # G4 可用区：us-central1-b/c/f、us-west1-a/b/c、europe-west4-b 等，以控制台为准
export NAME=g4-glm53
```

## 1. 建机器

```bash
PROJECT=$PROJECT ZONE=$ZONE NAME=$NAME ./scripts/create-vm.sh
```

关键参数：

| 参数 | 值 | 为什么 |
|---|---|---|
| `--machine-type` | `g4-standard-384` | 8 卡；模型 FP8 权重 306 GiB，至少要 4 张 96 GB 卡，TP8 才有足够 KV 空间 |
| `--image-family` | `common-cu129-ubuntu-2404-nvidia-580` | 自带 NVIDIA 580 驱动，SM120 需要 570 以上 |
| `--boot-disk-size` | `2000GB` | 权重 306 GiB + Docker 镜像约 30 GB，留余量 |
| `--boot-disk-type` | `hyperdisk-balanced` | G4 只支持 Hyperdisk |
| `--provisioning-model` | `SPOT` | 测试省钱；生产改按需 |

**验证**：

```bash
gcloud compute ssh $NAME --zone $ZONE --project $PROJECT --command "nvidia-smi -L"
```

应列出 8 行 `NVIDIA RTX PRO 6000 Blackwell Server Edition`。

## 2. 装 Docker 和 NVIDIA Container Toolkit

DLVM 镜像不带 Docker，需要自己装。以下在 VM 上执行：

```bash
curl -fsSL https://get.docker.com | sudo sh

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**验证**：`sudo docker run --rm --gpus all ubuntu nvidia-smi -L` 能看到 8 张卡。

> DLVM 镜像已经预装并 **hold** 住了 `nvidia-container-toolkit 1.19.1`。上面的 `apt-get install` 可能报
> `Depends: nvidia-container-toolkit-base (= 1.20.1-1) but 1.19.1-1 is to be installed` / `held broken packages`，
> **不影响使用**，只要 `nvidia-ctk runtime configure` 和上面的验证通过就行。另外别把其它包和它写在同一条 `apt-get install` 里，会被一起拖失败。

## 3. 拉镜像（可与第 4 步并行）

```bash
sudo docker pull vllm/vllm-openai:nightly
sudo docker run --rm --gpus all --entrypoint python3 vllm/vllm-openai:nightly \
  -c "import vllm; print(vllm.__version__)"
```

> 生产环境请记下 `sudo docker inspect --format '{{index .RepoDigests 0}}' vllm/vllm-openai:nightly` 的 digest，
> 之后用 `vllm/vllm-openai@sha256:...` 固定版本。本次测试用的 digest 见 [results/](results/)。

## 4. 下载模型

两种来源，**推荐 4B**：第一次从 HuggingFace 下（4A），顺手放进**和 VM 同区域**的 GCS 桶，以后都从桶里拷（4B）。

| 来源 | 实测耗时（306 GiB） | 实测速度 |
|---|---|---|
| 4A HuggingFace | 约 2.5 分钟（含装 uv） | 约 2 GB/s |
| 4B 同区域 GCS 桶（us-central1 → us-central1-b） | **49 秒** | **6.3 GiB/s** |
| 反向：本地上传到桶 | 77 秒 | 约 4 GiB/s |

### 4A. 从 HuggingFace 下载

```bash
# DLVM 的 Ubuntu 24.04 镜像里 python3 没带 pip，用 uv 建一个独立环境
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
uv venv ~/hfenv && uv pip install --python ~/hfenv/bin/python "huggingface_hub[hf_xet]"
sudo mkdir -p /data && sudo chown $USER /data
HF_XET_HIGH_PERFORMANCE=1 ~/hfenv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('zai-org/GLM-5.3-Flash', local_dir='/data/GLM-5.3-Flash', max_workers=32)"
du -sh /data/GLM-5.3-Flash
```

### 4B. 从同区域 GCS 桶拷（推荐）

桶要和 VM 在同一区域（本次桶在 `us-central1`，VM 在 `us-central1-b`），跨区域会慢且收出口流量费。
VM 用默认服务账号 + `--scopes cloud-platform` 即可，服务账号需要桶的 `roles/storage.objectViewer`（上传需 `objectAdmin`）。

```bash
export MODEL_URI=gs://<your-bucket>/models/zai-org/GLM-5.3-Flash

# 第一次：4A 下完后上传一份（.cache 是 huggingface_hub 的元数据，不用传）
gcloud storage rsync -r /data/GLM-5.3-Flash $MODEL_URI --exclude="^\.cache/"

# 以后：直接从桶拷
sudo mkdir -p /data && sudo chown $USER /data
gcloud storage cp -r $MODEL_URI /data/
```

`gcloud storage` 默认就会并行分片下载，不需要额外调参。

**验证**：`du` 约 306 GiB（HuggingFace 上全部文件合计 305.8 GiB）。实测下载速度约 2 GB/s（开头 60 秒下了 115 GB），整份权重约 2.5 分钟。

## 5. 启动服务

```bash
sudo docker run -d --name glm --gpus all --shm-size=64g --ipc=host --network host \
  -v /data/GLM-5.3-Flash:/model \
  vllm/vllm-openai:nightly \
  --model /model --served-model-name glm \
  --tensor-parallel-size 8 \
  --max-model-len 131072 --max-num-seqs 256 \
  --gpu-memory-utilization 0.85 --kv-cache-dtype fp8 \
  --tool-call-parser glm47 --reasoning-parser glm45 \
  --port 8000

sudo docker logs -f glm      # 看加载进度
```

参数说明：

| 参数 | 说明 |
|---|---|
| `--tensor-parallel-size 8` | 8 卡切一个副本 |
| `--max-model-len 131072` | 本次压测开 128K。要 1M 改成 `1048576`，KV 池需求更大 |
| `--max-num-seqs 256` | 原作者报告 1024 时 Mamba（KDA 状态）缓存 OOM，512 是上限 |
| `--kv-cache-dtype fp8` | SM120 的 sparse MLA 后端要求 `fp8_ds_mla` 格式 KV |
| `--tool-call-parser glm47` / `--reasoning-parser glm45` | 官方 recipe 给的 GLM 系解析器 |
| 不加 `--speculative-config` | **MTP 先别开**，nightly 原生 SM120 后端上接受率为 0% |

**验证**：日志里依次出现这几行（括号里是本次实测值）：

```
Using FLASHINFER_MLA_SPARSE_SM120 attention backend
Using fp8_ds_mla KV cache format for FLASHINFER_MLA_SPARSE_SM120 backend.
Model loading took 39.38 GiB memory and 31.5 seconds        （每卡）
Available KV cache memory: 38.14 GiB                         （每卡）
GPU KV cache size: 4,441,368 tokens, Maximum concurrency for 131,072 tokens per request: 33.88x
init engine (profile, create kv cache, warmup model) took 318.07 s
```

预热期间会刷 `No available shared memory broadcast block found in 60 seconds`，是 JIT 编译慢导致的提示，**不是挂了**，等着就行。最后出现 `Application startup complete`。

```bash
curl -s localhost:8000/v1/models
```

## 6. 冒烟测试

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm",
  "messages": [{"role": "user", "content": "用一句话介绍香港。"}],
  "max_tokens": 200
}'
```

**注意**：GLM-5.3-Flash 默认先思考再回答。上面这条请求 200 个 token 全用在思考上（`completion_tokens_details.reasoning_tokens: 200`），`content` 为空，**这是正常的**，不是服务坏了。两种处理：

- 调大 `max_tokens`（思考类请求建议 2,000 以上）；
- 或者关掉思考，直接出答案：

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm",
  "messages": [{"role": "user", "content": "用一句话介绍香港。"}],
  "max_tokens": 200,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

实测返回：「香港是一座融合东西方文化的国际大都市，以其繁华的金融中心地位、独特的摩天大楼天际线和"东方之珠"的魅力闻名于世。」

## 7. 压测

```bash
for cfg in "1024 512 1" "1024 512 4" "1024 512 16" "1024 512 64" \
           "8192 512 1" "8192 512 16" "32768 256 1"; do
  set -- $cfg
  n=$(( $3 * 4 )); [ $n -lt 8 ] && n=8
  sudo docker exec glm vllm bench serve \
    --backend openai-chat --endpoint /v1/chat/completions \
    --model glm --tokenizer /model \
    --dataset-name random --random-input-len $1 --random-output-len $2 \
    --max-concurrency $3 --num-prompts $n --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    | tee bench-in$1-out$2-c$3.txt
done
```

看每个文件里的 `Output token throughput`、`Mean TTFT`、`Mean TPOT`。

### 7.1 8K 输入 / 1K 输出并发扫描

```bash
# 服务起好后执行；结果写到 ~/g4glm/8k1k/，每档一个 .txt（摘要）和一个 .json（逐请求明细）
bash scripts/bench-8k1k.sh
# 只跑部分档位：CONCURRENCY="1 16 64" bash scripts/bench-8k1k.sh
```

| 参数 | 值 | 说明 |
|---|---|---|
| `--random-input-len` / `--random-output-len` | 8192 / 1024 | 随机 token，不命中前缀缓存 |
| `--max-concurrency` | 1 2 4 8 16 32 64 128 | 同时在飞的请求数 |
| `--num-prompts` | 并发 × 4，至少 8 | 每档跑 4 轮，摊掉首轮冷启动 |
| `--ignore-eos` | 开 | 每条都生成满 1,024 个 token，数字可比 |
| `--seed` | 等于并发数 | 每档数据固定，可复现 |

整轮扫描约 30 分钟（实测见 README 第 5.2 节）。

## 8. 收尾

```bash
sudo docker rm -f glm
# 不用了记得删机器，Spot 也按小时计费：
gcloud compute instances delete $NAME --zone $ZONE --project $PROJECT
```

## 9. PD 分离：1P1D（两台机器）

一台机器跑 prefill（P），另一台跑 decode（D），各自 TP8。KV 由 vLLM 的 `NixlConnector` 从 P 传到 D；
G4 跨机没有 RDMA，NIXL 的 UCX 后端走 **TCP**。镜像里已带 NIXL 1.5.2，并且已支持 KDA/Mamba 这类混合模型的状态传输，不用另装。

脚本都在 [`scripts/pd/`](scripts/pd/)：

| 脚本 | 在哪台跑 | 作用 |
|---|---|---|
| `setup-node.sh`（在 `scripts/`） | 每台 | 装 Docker、拉镜像、从桶拷权重，不起服务 |
| `pd/serve.sh` | P、D 各一次 | `ROLE=prefill` 起 `glm-p`（:8100，kv_producer）；`ROLE=decode` 起 `glm-d`（:8200，kv_consumer） |
| `pd/proxy.sh` | P | 起代理 `glm-proxy`（:8000）。代理先把请求发给 P 只算 prefill（`max_tokens=1`），拿到 KV 位置后转给 D 生成 |
| `pd/toy_proxy_server.py` | — | vLLM 官方 NIXL 示例代理（`tests/v1/kv_connector/nixl_integration`，commit `76c6f3f2a`），原样拷贝 |
| `pd/bench.sh` | P | 通过代理跑 8K / 1K 并发扫描 |
| `pd/run-1p1d.sh` | P | 一键：等 P/D 就绪 → 起代理 → 冒烟 → 扫描 |

### 9.1 准备第二台机器

```bash
NAME=g4-glm53-d ./scripts/create-vm.sh
# 在新机器上：
MODEL_URI=gs://<your-bucket>/models/zai-org/GLM-5.3-Flash bash setup-node.sh
```

两台机器要在**同一个 zone、同一个 VPC**，内网互通（默认 VPC 的 `default-allow-internal` 规则就够）。

测一下两台之间的带宽（本次实测）：

```bash
# D 上：iperf3 -s -D      P 上：
iperf3 -c <D_IP> -P 16 -t 10     # 实测 173 Gbit/s（16 条流）
iperf3 -c <D_IP> -P 1 -t 5       # 实测 49.7 Gbit/s（单条流）
```

### 9.2 起服务

```bash
# P 机器
ROLE=prefill bash pd/serve.sh
# D 机器
ROLE=decode bash pd/serve.sh
```

关键配置（`serve.sh` 里）：

| 项 | 值 | 说明 |
|---|---|---|
| `--kv-transfer-config` | `{"kv_connector":"NixlConnector","kv_role":"kv_producer"/"kv_consumer","kv_load_failure_policy":"fail"}` | P 是生产者、D 是消费者 |
| `VLLM_NIXL_SIDE_CHANNEL_HOST` | 本机内网 IP | NIXL 握手地址，**跨机必须设**，否则默认 localhost 对方连不上 |
| `VLLM_NIXL_SIDE_CHANNEL_PORT` | 5600 | 握手端口 |
| `UCX_TLS` / `UCX_NET_DEVICES` | `all` / `all` | 让 UCX 自己选；G4 上会落到 TCP + cuda_copy |
| 其余参数 | 与单机 TP8 完全相同 | 方便和单机结果对比 |

日志里出现 `NIXL is available` 说明连接器加载成功。

### 9.3 代理、冒烟、压测

```bash
# P 机器
P=<P_IP>:8100 D=<D_IP>:8200 bash pd/run-1p1d.sh
```

或分步：

```bash
PREFILL="<P_IP>:8100" DECODE="<D_IP>:8200" bash pd/proxy.sh
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm","messages":[{"role":"user","content":"用一句话介绍香港。"}],"max_tokens":200,"chat_template_kwargs":{"enable_thinking":false}}'
TAG=1p1d bash pd/bench.sh
```

客户端只认代理的 `:8000`，接口和单机完全一样。

## 排障

| 现象 | 处理 |
|---|---|
| 启动报 `pe_dim must be 64 for fp8_ds_mla` | 镜像太旧，没有 NoPE 修复（vLLM PR #55738）。换更新的 nightly |
| 启动时后端选错、偶发崩溃 | 原作者在社区镜像上要设 `FLASHINFER_MLA_SPARSE_SM90=1`；nightly 上此变量已不生效，后端自动选择 |
| 预热阶段 GPU 100% 卡住不动 | FlashInfer autotune 卡死，加 `--no-enable-flashinfer-autotune`（社区 TP2 报告） |
| `--max-num-seqs 1024` 时 OOM | KDA 状态池（Mamba cache）随并发线性涨，降到 512 以下 |
| `No module named pip` | DLVM Ubuntu 24.04 镜像没带 pip，按第 4 步用 uv |
| Spot 机被抢占 | `gcloud compute instances start $NAME`；权重在系统盘上不丢 |
