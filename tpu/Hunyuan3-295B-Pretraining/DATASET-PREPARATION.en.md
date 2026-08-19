> 🌐 [中文](DATASET-PREPARATION.md) | **English**

# Tencent Hunyuan 3 (295B-A21B) Pre-training Real Dataset Preparation & Engineering Guide

This document records the end-to-end process of preparing production-grade pre-training datasets for Hunyuan 3 (295B-A21B) on MaxText / TPU v7 (Ironwood), along with hardware-validated benchmarks. It provides two views:
1. **5-Minute Quickstart (Minimal Step-by-Step Production Runbook)**
2. **Comprehensive Architecture Deep-Dive (Design Principles, Format Selection, Grain Communication-Free Sharding, Benchmarks & Lessons Learned)**

---

## ⚡ Minimal Step-by-Step Production Runbook

### Step 1: Install Format Conversion Dependencies
```bash
python3 -m pip install -q pyarrow array-record huggingface_hub
```

### Step 2: Concurrently Download & Bake Full 10BT Dataset (14 Shards, 9.69M Documents, `group_size:1` + `tf.train.Example`)
```bash
python3 ~/gpu-tpu-pedia/tpu/Hunyuan3-295B-Pretraining/maxtext-hunyuan3/download_and_convert_full_10bt.py
```

### Step 3: Fast Sync to GCS Bucket (199 ArrayRecord Shards, 17.12 GiB)
```bash
gcloud storage rsync -r \
  ~/gpu-tpu-pedia/tpu/Hunyuan3-295B-Pretraining/data/fineweb-edu-arrayrecord/ \
  gs://your-bucket/datasets/hunyuan3-pretrain/train/fineweb-edu/
```

### Step 4: Launch Real-Data Pre-training on TPU v7 (64 Chips) (Native 120832 Vocab)
```bash
PLATFORM=v7 GCS_STAGE=gs://your-bucket/hy3 \
IMAGE=us-docker.pkg.dev/PROJECT/gcr.io/your-maxtext-latest:runner \
STEPS=1000 PDBS=13 \
bash ~/gpu-tpu-pedia/tpu/Hunyuan3-295B-Pretraining/maxtext-hunyuan3/run.sh real-data-prod \
  dataset_type=grain \
  grain_train_files="gs://your-bucket/datasets/hunyuan3-pretrain/train/fineweb-edu/*.arrayrecord" \
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

## 📊 Hardware Benchmarks Comparison (TPU v7 64 Chips / 128 Devices)

| Metric | Synthetic Baseline | Real Data (`PDBS=12`) | **Real Data (`PDBS=13` Peak)** |
|---|---|---|---|
| **Dataset** | Dummy Synthetic | **FineWeb-Edu 10BT** | **FineWeb-Edu 10BT (9.692M Documents)** |
| **Vocab Configuration** | 120,832 | **Native 120,832** | **Native 120,832** |
| **Storage Format** | In-Memory | GCS ArrayRecord (`group_size:1`) | **GCS ArrayRecord (`group_size:1`, `zstd`)** |
| **Sequence Length** | 4096 | 4096 (Sequence Packing) | **4096 (Sequence Packing)** |
| **Steady-State Step Time** | 20.61 s | **20.61 s** | **22.13 s** |
| **TFLOP/s / Device** | 331.1 TFLOP/s | **331.10 TFLOP/s** | **334.08 TFLOP/s** ⚡ (+0.91%) |
| **TFLOP/s / Chip** | 662.2 TFLOP/s | **662.20 TFLOP/s** | **668.16 TFLOP/s** 🚀 |
| **MFU (Denominator 2307)** | 28.70% | **28.71%** | **28.96%** |
| **Total Cluster Throughput** | 305,248 tok/s | **305,195 tok/s** | **307,925 tok/s** |
| **Loss Behavior** | — | 13.433 → 13.434 (LR=0 Baseline) | **13.433 → 9.848 Smooth Monotonic Convergence** (30 steps consuming ~100M tokens, lm_loss: 12.211 → 8.632, PPL: 201k → 7.1k) 📉 |
| **Data I/O Stall** | 0 ms | **0 ms (Overlapped)** | **0 ms (Overlapped)** |

---

## 📖 Deep-Dive Architecture Guide

### 1. Dataset Selection: Why FineWeb-Edu 10BT?
- **Global Deterministic Shuffle**: Grain's deterministic permutation requires a full 10-billion token base (9.69M articles) to achieve mathematically rigorous, uniform sampling.
- **Matched Step Scale**: 64 chips consume ~3M tokens per step; 10BT provides ample headroom for 3000+ verification steps.

### 2. Why Offline ArrayRecord Baking is Mandatory
- **JSONL / CSV**: Sequential scan only; CPU bound.
- **Parquet**: Columnar format introduces heavy decompression & row assembly overhead on Host CPUs.
- **ArrayRecord**: Row-oriented chunked compression with index table; enables C++ zero-copy direct reads from GCS.

### 3. GCS Storage Layout Specification

```
# Dataset Bucket (Pure Corpus Shards)
gs://your-bucket/datasets/hunyuan3-pretrain/
└── train/
    └── fineweb-edu/
        ├── fineweb_edu_train_part00-0000.arrayrecord
        ├── ... (199 shards in total, 17.12 GiB)
        └── fineweb_edu_train_part13-0003.arrayrecord

# Model Assets Bucket (Tokenizer & Model Config)
gs://your-bucket/hy3/
└── tokenizer/
    ├── config.json
    ├── generation_config.json
    ├── tokenizer.json
    └── tokenizer_config.json
```

---

## 🛠️ Lessons Learned & Gotchas

1. **ArrayRecordWriter Mandatory Option (`group_size:1`)**: Grain random sampling strictly requires `group_size:1`. Default 65536 causes immediate loader rejection.
2. **Data Encoding Protocol (`tf.train.Example`)**: MaxText's Grain pipeline deserializes `maxtext.input_pipeline.protos.Example`. Records must be encoded as binary Protobuf with `'text'` bytes field.
3. **Tokenizer Vocab Out-of-Bounds NaN**: Hunyuan 3 defaults to `vocab_size: 120832`. Using an external 128K tokenizer causes out-of-bounds cross-entropy loss evaluation $\log(0) = -\infty \to \text{NaN}$. Solution: use native 120,832 vocab with embedded direct loader.
4. **Time-based Priority Scheduling (`priority-dev` + `medium`)**: During weekday 4am - 4pm HKT window, `priority-dev` namespace provides top preemption and scheduling priority, reducing TPU init wait from minutes to under 20s.
