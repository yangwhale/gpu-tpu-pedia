#!/bin/bash
# ALModel 8B Pretraining Script
# Architecture: ALModel (MLA/KDA + MoE)
# Dataset: Local JSONL /models/dataset/oscar-en-10k.jsonl

set -e

# ============================================================================ 
# [TODO] 1. Tokenizer Path
# ============================================================================ 
# Must cover 163,840 vocabulary (tokenizer.model or tokenizer.json)
TOKENIZER_PATH="/ant-pretrain/src/MaxText/assets/my_tokenizer/"

if [[ -z "$TOKENIZER_PATH" ]]; then
    echo "❌ Error: Please set TOKENIZER_PATH in the script!"
    exit 1
fi

# ============================================================================ 
# 2. Basic Environment Config (GCS Bucket & Run Name)
# ============================================================================ 
BASE_OUTPUT_DIR=${GCS_BUCKET:-"gs://ant-pretrain-code/pretrain/dev"}
RUN_NAME=${RUN_NAME:-"al-model-8b-$(date +%Y%m%d)"}
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}"

# ============================================================================ 
# 3. JAX Multi-node Config (Auto-detect)
# ============================================================================ 
if [[ -n "$TPU_PROCESS_ADDRESSES" ]]; then
    export JAX_COORDINATOR_ADDRESS=$(echo $TPU_PROCESS_ADDRESSES | cut -d',' -f1)
    echo "🌐 Multi-node detection: Coordinator -> $JAX_COORDINATOR_ADDRESS"
    echo "   Worker ID: $TPU_WORKER_ID"
    echo "   TPU Topology: $TPU_TOPOLOGY"
else
    echo "⚠️ TPU_PROCESS_ADDRESSES not detected, assuming single machine."
fi

# ============================================================================ 
# 4. Dataset Config (Local JSONL Mode)
# ============================================================================ 
DATASET_PATH="/models/dataset/oscar-en-10k/oscar-en-10k.jsonl"
DATASET_TYPE="hf"
HF_LOADER_TYPE="json"
JSON_TEXT_KEY="text" 

if [[ ! -f "$DATASET_PATH" ]]; then
    echo "⚠️ Warning: Dataset file not found at: $DATASET_PATH"
fi

# ============================================================================ 
# 5. Training Hyperparameters
# ============================================================================ 
# Model Definition (Corresponds to src/MaxText/configs/models/al_model.yml)
MODEL_NAME="al_model"
CONFIG_FILE="src/MaxText/configs/base.yml"

# Training Steps and Batch Size
STEPS=10
PER_DEVICE_BATCH_SIZE=4        # Total Batch = 64 devices * 4 = 256
MAX_TARGET_LENGTH=8192

# Optimizer (Muon)
OPT_TYPE="muon"
LEARNING_RATE=5e-4
WARMUP_STEPS_FRACTION=0.1
MUON_WEIGHT_DECAY=0.1

# ============================================================================ 
# 6. Start Training Command
# ============================================================================ 
echo "========================================================"
echo "🚀 Starting ALModel 8B Training"
echo "   Model Arch : ALModel (MLA + MoE)"
echo "   Output Dir : $OUTPUT_DIR"
echo "   Tokenizer  : $TOKENIZER_PATH"
echo "   Dataset    : $DATASET_PATH"
echo "========================================================"

python3 -m MaxText.train "$CONFIG_FILE" \
    model_name=$MODEL_NAME \
    override_model_config=true \
    run_name=$RUN_NAME \
    base_output_directory=$BASE_OUTPUT_DIR \
    \
    `# --- Dataset Loading (JSONL) ---` \
    dataset_type=$DATASET_TYPE \
    hf_path=$HF_LOADER_TYPE \
    hf_train_files=$DATASET_PATH \
    dataset_name=$JSON_TEXT_KEY \
    \
    `# --- Tokenizer ---` \
    tokenizer_path=$TOKENIZER_PATH \
    vocab_size=163840 \
    \
    `# --- Training Parameters ---` \
    steps=$STEPS \
    per_device_batch_size=$PER_DEVICE_BATCH_SIZE \
    max_target_length=$MAX_TARGET_LENGTH \
    \
    `# --- Optimizer (Muon) ---` \
    opt_type=$OPT_TYPE \
    learning_rate=$LEARNING_RATE \
    warmup_steps_fraction=$WARMUP_STEPS_FRACTION \
    adam_weight_decay=0.1 \
    muon_weight_decay=$MUON_WEIGHT_DECAY \
    muon_consistent_rms=0.2 \
    cosine_learning_rate_final_fraction=0.1 \
    learning_rate_schedule_steps=-1 \
    \
    `# --- Parallelism Strategy ---` \
    ici_fsdp_parallelism=2 \
    ici_tensor_parallelism=2 \
    ici_context_parallelism=2 \
    dcn_fsdp_parallelism=1 \
    dcn_tensor_parallelism=1 \
    \
    `# --- Performance Optimization ---` \
    remat_policy=full \
    \
    `# --- System Config ---` \
    enable_checkpointing=true \
    checkpoint_period=5 \
    async_checkpointing=True \
    gcs_metrics=false \
    save_config_to_gcs=false \
    packing=false

echo "✅ Training finished (or submitted). Check GCS for logs."
