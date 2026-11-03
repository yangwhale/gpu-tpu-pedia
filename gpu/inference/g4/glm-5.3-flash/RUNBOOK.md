# RUNBOOK — 在 G4 上跑 GLM-5.3-Flash（vLLM · FP8 · TP8）

从零开始，按顺序执行。每一步都写了「做完应该看到什么」。
一键版：第 1 步建好机器后，把 [`scripts/run-all.sh`](scripts/run-all.sh) 拷上去用 `setsid nohup` 跑，它按顺序做完第 2–7 步。

总耗时参考：建机 1–2 分钟；装 Docker 约 1 分钟；下载 306 GiB 权重（待实测）；服务启动（待实测）；压测（待实测）。

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

## 3. 拉镜像（可与第 4 步并行）

```bash
sudo docker pull vllm/vllm-openai:nightly
sudo docker run --rm --gpus all --entrypoint python3 vllm/vllm-openai:nightly \
  -c "import vllm; print(vllm.__version__)"
```

> 生产环境请记下 `sudo docker inspect --format '{{index .RepoDigests 0}}' vllm/vllm-openai:nightly` 的 digest，
> 之后用 `vllm/vllm-openai@sha256:...` 固定版本。本次测试用的 digest 见 [results/](results/)。

## 4. 下载模型

```bash
python3 -m pip install --break-system-packages -U "huggingface_hub[hf_xet]"
sudo mkdir -p /data && sudo chown $USER /data
HF_XET_HIGH_PERFORMANCE=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('zai-org/GLM-5.3-Flash', local_dir='/data/GLM-5.3-Flash', max_workers=32)"
du -sh /data/GLM-5.3-Flash
```

**验证**：`du` 约 306 GiB（HuggingFace 上全部文件合计 305.8 GiB）。

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

**验证**：日志出现 `Using FLASHINFER_MLA_SPARSE_SM120 attention backend`，最后出现 `Application startup complete`。

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

应返回通顺的中文回答（`reasoning_content` 字段里是思考过程）。

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

## 8. 收尾

```bash
sudo docker rm -f glm
# 不用了记得删机器，Spot 也按小时计费：
gcloud compute instances delete $NAME --zone $ZONE --project $PROJECT
```

## 排障

| 现象 | 处理 |
|---|---|
| 启动报 `pe_dim must be 64 for fp8_ds_mla` | 镜像太旧，没有 NoPE 修复（vLLM PR #55738）。换更新的 nightly |
| 启动时后端选错、偶发崩溃 | 原作者在社区镜像上要设 `FLASHINFER_MLA_SPARSE_SM90=1`；nightly 上此变量已不生效，后端自动选择 |
| 预热阶段 GPU 100% 卡住不动 | FlashInfer autotune 卡死，加 `--no-enable-flashinfer-autotune`（社区 TP2 报告） |
| `--max-num-seqs 1024` 时 OOM | KDA 状态池（Mamba cache）随并发线性涨，降到 512 以下 |
| Spot 机被抢占 | `gcloud compute instances start $NAME`；权重在系统盘上不丢 |
