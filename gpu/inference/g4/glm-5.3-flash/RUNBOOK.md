# RUNBOOK — 在 G4 上跑 GLM-5.3-Flash（vLLM · FP8 · TP8，以及 PD 分离 1P1D）

从零开始，**按顺序一步不差地执行**就能复现本报告的全部数字。每一步都写了「做完应该看到什么」。

文中所有尖括号 `<...>` 都要换成你自己的值：

| 占位符 | 含义 | 例子 |
|---|---|---|
| `<your-project>` | 你的 GCP 项目 ID | `my-gpu-project` |
| `<your-bucket>` | 你自己的 GCS 桶（可选，用来缓存权重） | `my-models-uc1` |
| `<P_IP>` / `<D_IP>` | PD 分离时两台机器的内网 IP | `10.0.0.2` |

总耗时参考（本次实测）：

| 步骤 | 耗时 |
|---|---|
| 建 Spot 机器 | 1–2 分钟 |
| 装 Docker + 拉 vLLM 镜像（并行） | 约 1–2 分钟 |
| 下载 306 GiB 权重 | HuggingFace 约 2.5 分钟；同区域 GCS 桶 49 秒 |
| 服务启动 | 约 6.5 分钟（权重加载 31 s + 编译、预热、CUDA graph 共 318 s） |
| 基础压测 7 个场景 | 约 8 分钟 |
| 8K / 1K 并发扫描 8 档 | 约 30 分钟 |

**省事的一键版**：做完第 1 步后，把 [`scripts/run-all.sh`](scripts/run-all.sh) 拷到机器上跑，它会自动做完第 2–7 步（见第 1.3 节）。

---

## 0. 准备（在你自己的电脑上）

1. 一个有 G4 配额的 GCP 项目。`g4-standard-384` 一台需要 **8 张 `nvidia-rtx-pro-6000`** 的配额；用 Spot 时要的是 Spot（preemptible）配额。可在控制台「IAM 与管理 → 配额」里搜 `RTX PRO 6000` 查看。
2. 装好 Google Cloud CLI 并登录：

```bash
gcloud auth login
gcloud config set project <your-project>
gcloud services enable compute.googleapis.com
```

3. 把本仓库拉下来，进入本目录：

```bash
git clone https://github.com/yangwhale/gpu-tpu-pedia.git
cd gpu-tpu-pedia/gpu/inference/g4/glm-5.3-flash
```

4. 设几个后面反复用到的变量：

```bash
export PROJECT=<your-project>
export ZONE=us-central1-b      # 需是有 G4 的 zone，以控制台「可用区」列表为准
export NAME=g4-glm53
```

不需要 HuggingFace token：`zai-org/GLM-5.3-Flash` 是公开模型。

## 1. 建 Spot 机器

### 1.1 创建命令

下面这条和 [`scripts/create-vm.sh`](scripts/create-vm.sh) 完全一样，直接复制执行：

```bash
gcloud compute instances create $NAME \
  --project $PROJECT --zone $ZONE \
  --machine-type g4-standard-384 \
  --provisioning-model SPOT --instance-termination-action STOP \
  --maintenance-policy TERMINATE \
  --image-family common-cu129-ubuntu-2404-nvidia-580 \
  --image-project deeplearning-platform-release \
  --boot-disk-size 2000GB --boot-disk-type hyperdisk-balanced \
  --scopes cloud-platform \
  --metadata install-nvidia-driver=True
```

| 参数 | 值 | 为什么 |
|---|---|---|
| `--machine-type` | `g4-standard-384` | 8 卡 × 96 GB。FP8 权重 306 GiB，TP8 后每卡约 39 GiB，剩下的给 KV |
| `--provisioning-model SPOT` + `--instance-termination-action STOP` | Spot，被抢占时**停机不删盘** | 测试省钱；被抢占后 `start` 回来，权重还在 |
| `--image-family` | Deep Learning VM `common-cu129-ubuntu-2404-nvidia-580` | 预装 NVIDIA 驱动（本次实测 580.178.04）和 CUDA 12.9 |
| `--boot-disk-size 2000GB` / `hyperdisk-balanced` | 2 TB 系统盘 | 权重 306 GiB + vLLM 镜像解压后 35.5 GB，留余量 |
| `--scopes cloud-platform` | | 机器上能直接用 `gcloud storage` 读写你的桶 |

要按需（非 Spot）就去掉 `--provisioning-model` 和 `--instance-termination-action` 两个参数。
报 `ZONE_RESOURCE_POOL_EXHAUSTED` 说明这个 zone 暂时没有 Spot 容量，换一个有 G4 的 zone 再试。

### 1.2 验证

```bash
gcloud compute ssh $NAME --zone $ZONE --project $PROJECT --command "nvidia-smi -L"
```

应列出 8 行 `NVIDIA RTX PRO 6000 Blackwell Server Edition`。第一次 ssh 会生成密钥，偶尔要等十几秒才生效，报 `Permission denied (publickey)` 就重试一次。

### 1.3 把脚本拷上去（之后的步骤都在机器上执行）

```bash
gcloud compute scp --recurse scripts $NAME:~/ --zone $ZONE --project $PROJECT
gcloud compute ssh $NAME --zone $ZONE --project $PROJECT
# 现在已经在机器上了
```

一键版（自动做完第 2–7 步，日志在 `~/g4glm/run.log`，结果在 `~/g4glm/`）：

```bash
# 可选：权重已经放在你同区域的桶里，就设这一行，下载从 2.5 分钟变成 49 秒
export MODEL_URI=gs://<your-bucket>/models/zai-org/GLM-5.3-Flash
setsid nohup bash ~/scripts/run-all.sh > ~/g4glm-run.log 2>&1 < /dev/null &
tail -f ~/g4glm-run.log          # 看到 "=== STEP ALL_DONE" 即全部完成
```

想一步步看清楚，就按下面第 2–7 步手动执行。

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

**固定用本次测试的镜像 digest**，不要用浮动的 `nightly` 标签（它每天都变，复现出来的数字会不一样）：

```bash
export IMAGE=vllm/vllm-openai@sha256:88cc2118dab845fec1ff759c2fc8afa517cf8a668162ed8683b2f1b781372f57
sudo docker pull $IMAGE
sudo docker run --rm --gpus all --entrypoint python3 $IMAGE -c "import vllm; print(vllm.__version__)"
```

应打印 `0.33.0rc1.dev484+g76c6f3f2a`。这个 digest 是 2026-11-03 拉取的 `vllm/vllm-openai:nightly`。

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
  $IMAGE \
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

**注意**：GLM-5.3-Flash 默认以最高档（`max`）思考，再回答。上面这条请求 200 个 token 全用在思考上（`completion_tokens_details.reasoning_tokens: 200`，`finish_reason: length`），`content` 为空，**这是正常的**，不是服务坏了。两种处理：

- 调大 `max_tokens`（默认思考档位下建议 2,000 以上）；
- 或者把思考档位调低，直接出答案。这个模型的 chat template **没有** `enable_thinking` 开关（传了会被忽略），控制项是 `reasoning_effort`，取值 `low` / `high`，不传为 `max`：

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm",
  "messages": [{"role": "user", "content": "用一句话介绍香港。"}],
  "max_tokens": 2000,
  "reasoning_effort": "low"
}'
```

`reasoning_effort` 写在顶层（OpenAI 标准字段）或写成 `"chat_template_kwargs": {"reasoning_effort": "low"}` 都生效。本次实测 `low` 和 `high` 这条请求的 `reasoning_tokens` 都是 0，直接返回：「香港是一座融合东西方文化的国际化大都市，以繁华的金融中心、璀璨的城市夜景和独特的美食文化闻名于世。」

## 7. 压测

### 压测工具：`vllm bench serve`

**不用另外安装**：它就在 vLLM 镜像里，和服务端是同一个版本。上面第 5 步起的容器名叫 `glm`，直接 `docker exec` 进去调用即可。

如果想在**另一台机器**上压测（比如模拟真实客户端），在那台机器上用同一个镜像跑即可（需要一份模型目录给 tokenizer 用）：

```bash
sudo docker run --rm --network host -v /data/GLM-5.3-Flash:/model:ro --entrypoint vllm \
  vllm/vllm-openai@sha256:88cc2118dab845fec1ff759c2fc8afa517cf8a668162ed8683b2f1b781372f57 \
  bench serve --base-url http://<服务器IP>:8000 --model glm --tokenizer /model ...   # 其余参数同下
```

PD 分离的压测（第 9 节 `pd/bench.sh`）就是这么跑的。

`--tokenizer` 要指向一份模型目录（只用其中的 tokenizer 文件），用来按 token 数精确生成随机输入。

### 7.0 基础场景（7 个）


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
bash ~/scripts/bench-8k1k.sh
# 只跑部分档位：CONCURRENCY="1 16 64" bash ~/scripts/bench-8k1k.sh
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

前面那台（`$NAME`）当 P。在你自己的电脑上再建一台当 D，命令和第 1.1 节一样，只换名字：

```bash
export NAME_D=g4-glm53-d
NAME=$NAME_D PROJECT=$PROJECT ZONE=$ZONE bash scripts/create-vm.sh
gcloud compute scp --recurse scripts $NAME_D:~/ --zone $ZONE --project $PROJECT
gcloud compute ssh $NAME_D --zone $ZONE --project $PROJECT
```

在 D 上准备环境（装 Docker、拉同一个镜像、拷权重；不起服务）：

```bash
# 有桶就设 MODEL_URI，没有就先按第 4A 步从 HuggingFace 下到 /data/GLM-5.3-Flash
MODEL_URI=gs://<your-bucket>/models/zai-org/GLM-5.3-Flash bash ~/scripts/setup-node.sh
```

P 上如果还开着第 5 步的单机服务，先停掉：`sudo docker rm -f glm`。两台机器各自的内网 IP 用 `hostname -I | awk '{print $1}'` 查，记作 `<P_IP>`、`<D_IP>`。

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
ROLE=prefill bash ~/scripts/pd/serve.sh
# D 机器
ROLE=decode bash ~/scripts/pd/serve.sh
```

关键配置（`serve.sh` 里）：

| 项 | 值 | 说明 |
|---|---|---|
| `--kv-transfer-config` | `{"kv_connector":"NixlConnector","kv_role":"kv_producer"/"kv_consumer","kv_load_failure_policy":"fail"}` | P 是生产者、D 是消费者 |
| `VLLM_SSM_CONV_STATE_LAYOUT` | `DS` | **必须设**。GLM-5.3-Flash 有 KDA 线性注意力层（vLLM 按 Mamba 类状态管理），NIXL 传这类状态要求 DS 布局；不设会在引擎初始化末尾报 `3-read Mamba conv transfer requires DS conv state layout` 退出 |
| `VLLM_NIXL_SIDE_CHANNEL_HOST` | 本机内网 IP | NIXL 握手地址，**跨机必须设**，否则默认 localhost 对方连不上 |
| `VLLM_NIXL_SIDE_CHANNEL_PORT` | 5600 | 握手端口 |
| `UCX_TLS` / `UCX_NET_DEVICES` | `all` / `all` | 让 UCX 自己选；G4 上会落到 TCP + cuda_copy |
| 其余参数 | 与单机 TP8 完全相同 | 方便和单机结果对比 |

日志里出现 `NIXL is available` 说明连接器加载成功。

### 9.3 代理、冒烟、压测

```bash
# P 机器
P=<P_IP>:8100 D=<D_IP>:8200 bash ~/scripts/pd/run-1p1d.sh
# 结果在 ~/g4glm/pd-1p1d/，每档一个 .txt 和 .json
```

或分步：

```bash
PREFILL="<P_IP>:8100" DECODE="<D_IP>:8200" bash ~/scripts/pd/proxy.sh
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm","messages":[{"role":"user","content":"用一句话介绍香港。"}],"max_tokens":2000,"reasoning_effort":"low"}'
TAG=1p1d bash ~/scripts/pd/bench.sh
```

客户端只认代理的 `:8000`，接口和单机完全一样。

### 9.4 收尾

```bash
# 在你自己的电脑上
gcloud compute instances delete $NAME $NAME_D --zone $ZONE --project $PROJECT
```

## 排障

| 现象 | 处理 |
|---|---|
| 启动报 `pe_dim must be 64 for fp8_ds_mla` | 镜像太旧，没有 NoPE 修复（vLLM PR #55738）。换更新的 nightly |
| 启动时后端选错、偶发崩溃 | 原作者在社区镜像上要设 `FLASHINFER_MLA_SPARSE_SM90=1`；nightly 上此变量已不生效，后端自动选择 |
| 预热阶段 GPU 100% 卡住不动 | FlashInfer autotune 卡死，加 `--no-enable-flashinfer-autotune`（社区 TP2 报告） |
| `--max-num-seqs 1024` 时 OOM | KDA 状态池（Mamba cache）随并发线性涨，降到 512 以下 |
| PD 启动约 6 分钟后退出，报 `3-read Mamba conv transfer requires DS conv state layout. Set VLLM_SSM_CONV_STATE_LAYOUT=DS` | P、D 两边都加 `-e VLLM_SSM_CONV_STATE_LAYOUT=DS`（`pd/serve.sh` 已带） |
| 启动日志里大量 `CUDACachingAllocator ... memory allocation failed with OOM` | 显存 profile 阶段的正常警告，健康启动也有，**不是**失败原因；看 `RuntimeError` / `AssertionError` 那行 |
| 回复 `content` 为空、`finish_reason: length` | 默认 `max` 思考档把 `max_tokens` 用完了。调大 `max_tokens`，或传 `"reasoning_effort": "low"`；`enable_thinking` 对这个模型无效 |
| `No module named pip` | DLVM Ubuntu 24.04 镜像没带 pip，按第 4 步用 uv |
| Spot 机被抢占 | `gcloud compute instances start $NAME`；权重在系统盘上不丢 |
