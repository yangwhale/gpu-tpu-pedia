# Qwen3-235B-A22B TPU v7 (Ironwood) 快速上手指南

本指南指导如何在 Google Cloud TPU v7 共享集群（`bodaborg-tpu7x-nap`）上启动并运行 **Qwen3-235B-A22B** 预训练。

---

## 1. 硬件与架构基础

- **芯片型号**：TPU v7x (Ironwood)
- **拓扑规格**：`4x4x4`（16 台主机 / 64 芯片 / 128 devices）
- **每芯片规格**：2 devices / chip，峰值算力 2,307 TFLOPS/chip (BF16)
- **模型规格**：
  - 总参数量：235B（激活参数 22B）
  - 解码层数：94 层
  - 隐藏维度：4,096，单专家 MLP 维度：1,536
  - 专家数量：128（Top-8 路由），完美整除 FSDP 128 分片

---

## 2. 预训练 Golden 启动命令

### 2.1 🏆 FP8 + QAG 巅峰性能命令 (750.0 TFLOP/s / 32.51% MFU)

创建 `/tmp/tkcfg.py` 注入 monkeypatch，并启动：

```bash
cat << 'PYEOF' > /tmp/tkcfg.py
import os, dataclasses
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu as P
_TM, _TK, _TN = int(os.environ.get("TK_TM", 512)), int(os.environ.get("TK_TK", 2048)), int(os.environ.get("TK_TN", 1536))
_orig = P.PallasMosaicTpuRaggedDot._get_heuristics_config
def _patched(self, ba):
    c = _orig(self, ba)
    k, n = ba.arguments["rhs"].shape[-2], ba.arguments["rhs"].shape[-1]
    return dataclasses.replace(c, tile_m=_TM, tile_k=min(_TK, k), tile_n=min(_TN, n))
P.PallasMosaicTpuRaggedDot._get_heuristics_config = _patched
PYEOF

export TK_TM=512 TK_TK=2048 TK_TN=1536
export JAX_PLATFORMS=tpu,cpu TPU_STDERR_LOG_LEVEL=0 TF_CPP_MIN_LOG_LEVEL=0
export LIBTPU_INIT_ARGS='--xla_tpu_dvfs_p_state=7 \
  --xla_tpu_scoped_vmem_limit_kib=65472 \
  --xla_enable_async_all_gather=true \
  --xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
  --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
  --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_aggregator=true \
  --xla_tpu_use_tc_device_shape_on_sc=True \
  --xla_sc_disable_megacore_partitioning=True \
  --xla_tpu_enable_latency_hiding_layer_scheduler=true \
  --xla_tpu_scheduler_percent_shared_memory_limit=150 \
  --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true \
  --xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false'

cd /deps
python3 -c "
import sys, runpy
exec(open('/tmp/tkcfg.py').read())
sys.argv = ['train.py', 'src/maxtext/configs/base.yml',
  'model_name=qwen3-235b-a22b', 'run_name=qwen3-235b-fp8-qag-prod', 'base_output_directory=/tmp/qwen3out',
  'dataset_type=synthetic', 'enable_checkpointing=False', 'steps=10',
  'quantization=fp8_full', 'use_qwix_quantization=True',
  'weight_quantization_calibration_method=fixed,-224,224',
  'act_quantization_calibration_method=absmax',
  'bwd_quantization_calibration_method=absmax',
  'shard_exp_on_fsdp=True', 'use_tokamax_gmm=True',
  'dtype=bfloat16', 'weight_dtype=float32',
  'override_model_config=True', 'ici_fsdp_parallelism=-1', 'ici_tensor_parallelism=1',
  'megablox=True', 'sparse_matmul=True', 'scan_layers=True',
  'sa_block_q=2048', 'sa_block_kv=2048', 'sa_block_kv_compute=2048', 'sa_block_q_dkv=2048',
  'sa_block_kv_dkv=2048', 'sa_block_kv_dkv_compute=2048', 'sa_block_q_dq=2048', 'sa_block_kv_dq=2048',
  'use_max_logit_estimate=30',
  'remat_policy=custom', 'decoder_layer_input=offload', 'attention=flash',
  'allow_split_physical_axes=True', 'tokenizer_type=tiktoken',
  'tokenizer_path=src/maxtext/assets/tokenizer_llama3.tiktoken',
  'per_device_batch_size=13', 'max_target_length=4096', 'use_custom_sort_vjp=True',
  'sa_use_fused_bwd_kernel=True', 'use_tokamax_splash=True', 'out_proj=remat',
  'opt_type=adamw', 'mu_dtype=bfloat16', 'grad_dtype=bfloat16', 'use_iota_embed=True',
  'wi_tile_fwd_batch_seq=512', 'wi_tile_fwd_embed_dim=2048', 'wi_tile_fwd_mlp_dim=1536',
  'wi_tile_dlhs_batch_seq=512', 'wi_tile_dlhs_embed_dim=2048', 'wi_tile_dlhs_mlp_dim=1536',
  'wi_tile_drhs_batch_seq=512', 'wi_tile_drhs_embed_dim=2048', 'wi_tile_drhs_mlp_dim=1536',
  'wo_tile_fwd_batch_seq=512', 'wo_tile_fwd_embed_dim=2048', 'wo_tile_fwd_mlp_dim=1536',
  'wo_tile_dlhs_batch_seq=512', 'wo_tile_dlhs_embed_dim=2048', 'wo_tile_dlhs_mlp_dim=1536',
  'wo_tile_drhs_batch_seq=512', 'wo_tile_drhs_embed_dim=2048', 'wo_tile_drhs_mlp_dim=1536'
]
runpy.run_module('src.maxtext.trainers.pre_train.train', run_name='__main__')
"
```

### 2.2 FP8 absmax 生产命令 (718.4 TFLOP/s / 31.14% MFU)

```bash
cd /deps
python3 -m src.maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
  model_name=qwen3-235b-a22b run_name=qwen3-235b-fp8-v7 base_output_directory=/tmp/qwen3out \
  dataset_type=synthetic enable_checkpointing=False steps=10 \
  quantization=fp8_full use_qwix_quantization=True \
  weight_quantization_calibration_method=absmax \
  act_quantization_calibration_method=absmax \
  bwd_quantization_calibration_method=absmax \
  dtype=bfloat16 weight_dtype=float32 \
  override_model_config=True ici_fsdp_parallelism=-1 ici_tensor_parallelism=1 \
  megablox=True sparse_matmul=True scan_layers=True \
  sa_block_q=2048 sa_block_kv=2048 sa_block_kv_compute=2048 sa_block_q_dkv=2048 \
  sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=2048 sa_block_q_dq=2048 sa_block_kv_dq=2048 \
  use_max_logit_estimate=30 \
  remat_policy=custom decoder_layer_input=offload attention=flash \
  allow_split_physical_axes=True tokenizer_type=tiktoken \
  tokenizer_path=src/maxtext/assets/tokenizer_llama3.tiktoken \
  per_device_batch_size=13 max_target_length=4096 use_custom_sort_vjp=True \
  sa_use_fused_bwd_kernel=True use_tokamax_splash=True out_proj=remat \
  opt_type=adamw mu_dtype=bfloat16 grad_dtype=bfloat16 use_iota_embed=True \
  wi_tile_fwd_batch_seq=512 wi_tile_fwd_embed_dim=2048 wi_tile_fwd_mlp_dim=1536 \
  wi_tile_dlhs_batch_seq=512 wi_tile_dlhs_embed_dim=2048 wi_tile_dlhs_mlp_dim=1536 \
  wi_tile_drhs_batch_seq=512 wi_tile_drhs_embed_dim=2048 wi_tile_drhs_mlp_dim=1536 \
  wo_tile_fwd_batch_seq=512 wo_tile_fwd_embed_dim=2048 wo_tile_fwd_mlp_dim=1536 \
  wo_tile_dlhs_batch_seq=512 wo_tile_dlhs_embed_dim=2048 wo_tile_dlhs_mlp_dim=1536 \
  wo_tile_drhs_batch_seq=512 wo_tile_drhs_embed_dim=2048 wo_tile_drhs_mlp_dim=1536
```

### 2.3 BF16 极限峰值命令 (683.7 TFLOP/s / 29.63% MFU)

将上述命令中移除 `quantization=fp8_full` 相关的量化参数即可。

---

## 3. 监控与指标读数

稳态指标（Step 4+）预期：
- **FP8 + QAG (pdbs=13)**：
  - `seconds`: ~21.06 s
  - `TFLOP/s/device`: ~375.0 TFLOP/s
  - `TFLOP/s/chip` = 375.0 × 2 = **750.0 TFLOP/s**
  - `MFU`: **32.51%** (BF16 基准 2307) / **16.25%** (FP8 基准 4614)
  - `Tokens/s/device`: ~2,528.1 (全集群 323,597 Tokens/s)
- **FP8 absmax (pdbs=13)**：
  - `seconds`: ~21.99 s
  - `TFLOP/s/device`: ~359.2 TFLOP/s
  - `TFLOP/s/chip` = 359.2 × 2 = **718.4 TFLOP/s**
  - `MFU`: **31.14%** (BF16 基准 2307) / **15.57%** (FP8 基准 4614)
  - `Tokens/s/device`: ~2,421.5 (全集群 309,952 Tokens/s)
- **BF16 (pdbs=13)**：
  - `seconds`: ~23.11 s
  - `TFLOP/s/device`: ~341.8 TFLOP/s
  - `TFLOP/s/chip` = 341.8 × 2 = **683.7 TFLOP/s**
  - `MFU`: **29.63%** (BF16 基准 2307)
  - `Tokens/s/device`: ~2,304.6 (全集群 294,982 Tokens/s)
