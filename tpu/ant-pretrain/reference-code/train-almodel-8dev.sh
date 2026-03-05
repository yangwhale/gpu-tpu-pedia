#!/bin/bash
# ALModel Training Script for TPU v7 4-chip (8 devices, 单 host)
# Topology: 2x2x1, ICI: fsdp=2, tensor=2, context=1
# Tested: 2026-02-22, batch=1, seq=2048, HBM=52.73G/94.75G
set -e

echo "=== ALModel Training on TPU v7 4-chip (8 devices) ==="
echo "Start time: $(date)"
echo "Hostname: $(hostname)"

# --- Download ant-pretrain code from GCS ---
echo "=== Downloading ant-pretrain code ==="
mkdir -p /ant-pretrain
gsutil -m cp -r gs://chrisya-v7x-us-central1/ant-pretrain/* /ant-pretrain/

# --- Install ant-pretrain as package ---
echo "=== Installing ant-pretrain ==="
cd /ant-pretrain
pip install -e . 2>&1 | tail -5

# --- Prepare dataset ---
echo "=== Preparing dataset ==="
mkdir -p /models/dataset/oscar-en-10k
gsutil cp gs://chrisya-v7x-us-central1/ant-pretrain/test-dataset.jsonl /models/dataset/oscar-en-10k/oscar-en-10k.jsonl
echo "Dataset lines: $(wc -l < /models/dataset/oscar-en-10k/oscar-en-10k.jsonl)"

# --- Run training ---
echo "=== Starting ALModel Training ==="
echo "ICI: fsdp=2, tensor=2, context=1 (2x2x1=4 devices... wait, 8 devices)"
echo "Batch: per_device=1, seq_len=2048"

cd /ant-pretrain
python3 -m MaxText.train src/MaxText/configs/base.yml \
    model_name=al_model \
    override_model_config=true \
    run_name=almodel-4chip-test \
    base_output_directory=gs://chrisya-v7x-us-central1/almodel-training-output \
    dataset_type=hf \
    hf_path=json \
    hf_train_files=/models/dataset/oscar-en-10k/oscar-en-10k.jsonl \
    dataset_name=text \
    tokenizer_path=/ant-pretrain/src/MaxText/assets/my_tokenizer/ \
    vocab_size=163840 \
    steps=10 \
    per_device_batch_size=1 \
    max_target_length=2048 \
    opt_type=muon \
    learning_rate=5e-4 \
    warmup_steps_fraction=0.1 \
    adam_weight_decay=0.1 \
    muon_weight_decay=0.1 \
    cosine_learning_rate_final_fraction=0.1 \
    learning_rate_schedule_steps=-1 \
    ici_fsdp_parallelism=2 \
    ici_tensor_parallelism=2 \
    ici_context_parallelism=1 \
    dcn_fsdp_parallelism=1 \
    dcn_tensor_parallelism=1 \
    remat_policy=full \
    enable_checkpointing=true \
    checkpoint_period=5 \
    async_checkpointing=True \
    gcs_metrics=false \
    save_config_to_gcs=false \
    packing=false

echo "=== Training Complete ==="
echo "End time: $(date)"
