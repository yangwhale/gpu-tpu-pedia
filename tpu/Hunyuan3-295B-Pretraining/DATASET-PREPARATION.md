> 🌐 **中文** | [English](DATASET-PREPARATION.en.md)

# 腾讯混元 3（295B-A21B）预训练真实数据准备与工程实践

本文档记录如何为混元 3（Hunyuan 3 295B-A21B）在 MaxText / TPU v7 集群下准备生产级预训练数据集并完成真机实测验证。包含两套视图：
1. **五分钟快速上手（极简 Step-by-Step 生产操作卡）**
2. **深度全景指南（设计原理、格式选型、Grain 无通信分片、真机实测数据与踩坑台账）**

---

## ⚡ 极简 Step-by-Step 生产操作卡（复制即用）

### 步骤 1：安装格式转换依赖
```bash
python3 -m pip install -q pyarrow array-record huggingface_hub
```

### 步骤 2：一键并发下载并烘焙 10BT 完整数据集（14 个分片，969.2 万篇文档，`group_size:1` + `tf.train.Example`）
```bash
python3 ~/gpu-tpu-pedia/tpu/Hunyuan3-295B-Pretraining/maxtext-hunyuan3/download_and_convert_full_10bt.py
```

### 步骤 3：高速同步至 GCS 训练存储桶（共 199 个分片，17.12 GiB）
```bash
gcloud storage rsync -r \
  ~/gpu-tpu-pedia/tpu/Hunyuan3-295B-Pretraining/data/fineweb-edu-arrayrecord/ \
  gs://bodaborg-tpu7x-nap-us-central1/chrisya/datasets/hunyuan3-pretrain/train/fineweb-edu/
```

### 步骤 4：在 TPU v7（64 芯片）上启动真实数据预训练（官方 120832 原生词表）
```bash
PLATFORM=v7 GCS_STAGE=gs://bodaborg-tpu7x-nap-us-central1/chrisya/hy3 \
IMAGE=us-docker.pkg.dev/cloud-tpu-multipod-dev/gcr.io/chrisya-maxtext-latest:runner \
STEPS=1000 PDBS=13 \
bash ~/gpu-tpu-pedia/tpu/Hunyuan3-295B-Pretraining/maxtext-hunyuan3/run.sh real-data-prod \
  dataset_type=grain \
  grain_train_files="gs://bodaborg-tpu7x-nap-us-central1/chrisya/datasets/hunyuan3-pretrain/train/fineweb-edu/*.arrayrecord" \
  tokenizer_type="huggingface" \
  tokenizer_path="src/maxtext/assets/hunyuan3_tokenizer.json" \
  vocab_size=120832 \
  use_qwix_quantization=False \
  quantization="" \
  learning_rate=3e-5 \
  learning_rate_schedule_steps=100000 \
  warmup_steps_fraction=0.05 \
  gradient_clipping_threshold=1.0
```

---

## 📊 真机实测数据对照（TPU v7 64 芯片 / 128 Devices）

| 指标 | 合成数据基线（Synthetic Data） | 真实预训练数据 (`PDBS=12`) | **真实预训练数据 (`PDBS=13` 极限)** |
|---|---|---|---|
| **数据集** | Dummy Synthetic (无语义) | **FineWeb-Edu 10BT** | **FineWeb-Edu 10BT (969.2 万篇优质文档)** |
| **词表配置** | 120,832 | **官方原生 120,832** | **官方原生 120,832** |
| **存储格式** | 内存生成 | GCS ArrayRecord (`group_size:1`) | **GCS ArrayRecord (`group_size:1`, `zstd`)** |
| **序列长度** | 4096 | 4096 (在线 Sequence Packing) | **4096 (在线 Sequence Packing)** |
| **单步稳态耗时** | 20.61 s | **20.61 s** | **22.13 s** |
| **TFLOP/s / device** | 331.1 TFLOP/s | **331.10 TFLOP/s** | **334.08 TFLOP/s** ⚡ (+0.91%) |
| **TFLOP/s / chip** | 662.2 TFLOP/s | **662.20 TFLOP/s** | **668.16 TFLOP/s** 🚀 |
| **MFU (分母 2307)** | 28.70% | **28.71%** | **28.96%** |
| **整机 Token 吞吐** | 305,248 tok/s | **305,195 tok/s** | **307,925 tok/s** |
| **Loss 表现** | — | 13.433 → 13.434 (LR=0 稳态基线) | **13.433 → 9.848 完美平滑收敛** (30 步吃进 ~1 亿 Token，lm_loss: 12.211 → 8.632，PPL: 201k → 7.1k) 📉 |
| **数据 I/O 瓶颈** | 0 ms | **0 ms (完全掩盖在计算内)** | **0 ms (完全掩盖在计算内)** |

---

## 📖 深度全景指南

### 一、数据集选型：为什么锁定 FineWeb-Edu 10BT 全集？

对于中英双语 MoE 大模型（295B 总参数，21B 激活），开源界最强底座首选 **HuggingFace FineWeb-Edu 10BT**：
- **真正的确定性全局打散底座**：必须拥有完整的 100 亿 Token（9,692,101 篇高质量文章），Grain 才能在全局置换数学公式下实现真正均匀的随机采样。
- **单步吞吐匹配**：混元 3 64 芯片单步消耗约 300 万 Token，10BT 样本足够跑 3000+ 步完整收敛验证。

### 二、为什么必须在离线转为 ArrayRecord？

| 格式对比 | 存储原理 | TPU 多机寻址性能 | Host CPU 开销 | 评价 |
|---|---|---|---|---|
| **JSONL / CSV** | 行式纯文本 | 差（必须顺序全量扫描） | 极高（字符串解析与正则） | 仅限本地玩具测试 |
| **Parquet** | 列式压缩存储 | 中（跨列重组开销大） | 高（动态解压与行重组） | 适合 SQL 分析，不适合万卡流式训练 |
| **ArrayRecord** | **行式带索引块压缩** | **极高（毫秒级单条随机寻址）** | **极低（C++ 零拷贝直通）** | **TPU / Grain 官方唯一黄金标准** |

实测数据：
- 原始数据：FineWeb-Edu `sample/10BT/`（14 个分片 Parquet，共 28 GB，包含 969.2 万篇高质量完整文章）。
- 并发流式转换：4 线程流水线并发，流式批处理。
- 生成产物：**199 个 ArrayRecord 分片**（每片 50,000 条记录，Zstandard 压缩后总计 17.12 GiB）。
- GCS 上传吞吐：823.3 MiB/s，全量秒级入库。

### 三、GCS 目录结构规范

```
# 数据集存储桶（纯语料数据分片）
gs://bodaborg-tpu7x-nap-us-central1/chrisya/datasets/hunyuan3-pretrain/
└── train/
    └── fineweb-edu/
        ├── fineweb_edu_train_part00-0000.arrayrecord
        ├── ... (共 199 个分片，17.12 GiB)
        └── fineweb_edu_train_part13-0003.arrayrecord

# 模型资产存储桶（分词器与配置）
gs://bodaborg-tpu7x-nap-us-central1/chrisya/hy3/
└── tokenizer/
    ├── config.json
    ├── generation_config.json
    ├── tokenizer.json
    └── tokenizer_config.json
```

---

## 🛠️ 踩坑记录与关键经验 (Lessons Learned)

1. **ArrayRecordWriter 强制选项 (`group_size:1`)**：
   - Grain 库做多机随机索引抽样时，**强制要求 `group_size:1`**。默认 group_size=65536 会直接导致 `Grain requires group size 1 for good performance` 并拒绝读取。
   - 正确 options：`"group_size:1,zstd"`。
2. **数据编码协议 (Protobuf tf.train.Example)**：
   - MaxText 原生 Grain 数据流水线解码的是 `maxtext.input_pipeline.protos.Example`（等价于 `tf.train.Example`）。
   - 不能直接存 JSON 字符串，每条 record 必须编码为包含 `'text'` bytes 字段的 `tf.train.Example` 二进制 Protobuf。
3. **Parquet 批量迭代优化**：
   - 避免使用 Python 原生 `table["text"][i]` 单条索引；改用 `ParquetFile.iter_batches(batch_size=10000)`，处理速度提升 40 倍。
4. **Grain 随机读取 API**：
   - `ArrayRecordReader.num_records()` 获取总条数，读取单条或多条必须传入列表 `reader.read([index])`。
5. **GCS rsync 命令行工具**：
   - `gcloud storage rsync` 原生内置多线程并发，不要再加 `-m` 参数（`-m` 是旧版 `gsutil` 专属）。
6. **真实数据冷启动与数值稳定性（FP8 vs BF16）**：
   - 真实数据从零预训练（From Scratch）时，**严禁使用固定 Scale 量化（`fixed,-224,224`）**。随机初始化的真实文本 logits 与激活值动态范围较大，固定 224 截断会在 Step 1 反向求导时产生 NaN 溢出。
   - 预训练初跑建议先使用 **BF16 基线（`use_qwix_quantization=False`）** 或 **动态最大值量化（`absmax`）**。
7. **短步数压测下的 Warmup 塌陷陷阱（`learning_rate_schedule_steps`）**：
   - MaxText 的 `learning_rate_schedule_steps` 默认会直接继承总步数 `steps`。
   - 当为了快速验证性能将 `steps` 设得很小（如 `steps=6` 或 `steps=10`）时，`warmup_steps = int(steps * 0.1) = 0 ~ 1`。这会导致 Warmup 几乎完全失效，从 Step 1 起直接以 100% 峰值学习率进行大模型参数更新。
   - 正确做法：若仅做性能/数据流压测，显式指定 `learning_rate=0.0`；若做收敛验证，必须显式指定 `learning_rate_schedule_steps=100000`。
8. **Tokenizer 词表与模型 `vocab_size` 不匹配导致的越界 NaN 灾难**：
   - 混元 3 官方模型默认 `vocab_size: 120832`。
   - 若在预训练中直接挂载 LLaMA 3 的 tiktoken 分词器（`tokenizer_llama3.tiktoken`，词表大小为 `128256`），真实文本分词后会产生在 `[120832, 128256)` 区间的 Token ID。
   - 当这些越界 Token 作为 Target 计算 Cross Entropy 交叉熵损失时，由于超出 Logits 最后一维大小，直接索引越界导致 $\log(0) = -\infty$，反向传播时瞬间产生 NaN。
   - 正确解法：使用官方原生 120,832 词表，并以 `tokenizers.Tokenizer.from_file` 原生内嵌，零网络依赖且完全消除越界。
9. **时段优先队列调度策略（`priority-dev` + `medium`）**：
   - 在 Bodaborg TPU v7 集群中，平日 4am - 4pm HKT 期间 `priority-dev` 命名空间拥有最高调度优先级与资源抢占权。
   - 提交任务时显式指定 `priorityClassName: medium` + `namespace: priority-dev`，切片建立与 Pod 启动时间从数分钟缩短至 20 秒以内。
