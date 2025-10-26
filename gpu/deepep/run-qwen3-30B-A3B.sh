#!/bin/bash

export MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Megatron-LM"}
export MCORE_RELEASE_VERSION=${MCORE_RELEASE_VERSION:-"0.14"}


# # Training parameters
export DATA_CACHE_PATH="/scratch_data"
export OUTPUT_PATH="/scratch_data/output"
export DATA_PATH="/workspace/wikitext-converted/wikitext-converted_text_document"
export TOKENIZER_MODEL="/workspace/Qwen3-30B-A3B/"
export WANDB_BASE_URL=
export WANDB_API_KEY=
export WANDB_PROJECT=
export PROFILE=0 # whether to profile the model with nsys profile
export GBS=512
export MBS=32
export SEQ_LEN=4096
export MOE_GROUPED_GEMM=1
export PP=1
export VPP=1
export TP=1
export EP=8
export CP=1
export MOE_TOKEN_DISPATCHER=flex
export MOE_ENABLE_DEEPEP=1
export FULL_RECOMPUTE=1

export RUN_TIME=00:30:00
export COMMENT=

export CUDA_DEVICE_MAX_CONNECTIONS=1
GPU=$(nvidia-smi -q | grep 'Product Name' | head -n 1 | sed -r 's#.*NVIDIA (.*)#\1#g')

if [[ ! -d ${DATA_CACHE_PATH} ]]; then
    mkdir -p ${DATA_CACHE_PATH}
fi

# Profile command
if [[ ${PROFILE} -eq 1 ]]; then
    NSYS_PATH="${OUTPUT_PATH}/nsys"
    DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
    mkdir -p "${NSYS_PATH}"
    PROFILE_CMD="nsys profile --sample=none --cpuctxsw=none -t cuda,nvtx \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --cuda-memory-usage true \
        -f true -x true \
        -o ${NSYS_PATH}/${MODEL}-benchmarking-${DATETIME}"
    TRAINING_PARAMS="${TRAINING_PARAMS} --profile --profile-step-start 50 --profile-step-end 55 --profile-ranks 0 "
else
    PROFILE_CMD=""
fi

MEGATRON_ARGS=(
  # Distributed args
  --distributed-timeout-minutes 60
  --tensor-model-parallel-size ${TP}
  --pipeline-model-parallel-size ${PP}
#   --num-layers-per-virtual-pipeline-stage ${LAYERS_PER_VP}
  --expert-model-parallel-size ${EP}
  --context-parallel-size ${CP}
  --expert-tensor-parallel-size 1
  --use-distributed-optimizer 
  --overlap-grad-reduce 
  --overlap-param-gather 
  --no-create-attention-mask-in-dataloader 

  # Training args
  --use-mcore-models 
  --sequence-parallel 
  --use-flash-attn 
  --disable-bias-linear 
  --micro-batch-size ${MBS}
  --global-batch-size ${GBS}
  --train-samples 268554688
  --exit-duration-in-mins 230
  --manual-gc 
  --manual-gc-interval 5
  --cross-entropy-loss-fusion 
  --cross-entropy-fusion-impl te
  --enable-experimental 

  # Transformer Engine args
  --transformer-impl transformer_engine

  # Data args
  --data-cache-path ${DATA_CACHE_PATH}
  --tokenizer-type HuggingFaceTokenizer 
  --tokenizer-model ${TOKENIZER_MODEL}
  --data-path ${DATA_PATH}
  --split 99,1,0
#   --no-mmap-bin-files 
  --num-workers 12

  # Add network size args
  --untie-embeddings-and-output-weights 
  --position-embedding-type rope
  --rotary-percent 1.0
  --rotary-base 1000000
  --rotary-seq-len-interpolation-factor 1
  --normalization RMSNorm
  --swiglu 
  --norm-epsilon 1e-06
  --num-layers 48
  --hidden-size 2048
  --ffn-hidden-size 6144
  --num-attention-heads 32
  --group-query-attention 
  --num-query-groups 4
  --kv-channels 128
  --qk-layernorm 
  --seq-length ${SEQ_LEN}
  --max-position-embeddings 40960
  --make-vocab-size-divisible-by 1187

  # Add regularization args
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --clip-grad 1.0
  --weight-decay 0.1

  # Add learning rate args
  --lr-decay-samples 255126953
  --lr-warmup-samples 162761
  --lr 1.2e-4
  --min-lr 1.2e-5
  --lr-decay-style cosine
  --adam-beta1 0.9
  --adam-beta2 0.95

  # Add MoE args
  --num-experts 128
  --moe-ffn-hidden-size 768
  --moe-router-load-balancing-type aux_loss
  --moe-router-topk 8
#   --moe-router-pre-softmax
  --moe-aux-loss-coeff 1e-3
  --moe-token-dispatcher-type ${MOE_TOKEN_DISPATCHER}
  --moe-permute-fusion 
  --moe-router-dtype fp32
  --moe-router-fusion 

  # Add validation args
  --eval-iters 32
  --eval-interval 500

  # Add checkpointing args
  --finetune 
  --auto-detect-ckpt-format 
#   --load ${LOAD_PATH}
  --save ${OUTPUT_PATH}/checkpoints
  --save-interval 500
  --dist-ckpt-strictness log_all

  # Add initialization args
  --init-method-std 0.02

  # Add logging args
  --log-timers-to-tensorboard 
  --log-memory-to-tensorboard 
  --log-num-zeros-in-grad 
  --log-params-norm 
  --log-validation-ppl-to-tensorboard 
  --log-throughput 
  --log-interval 1
  --tensorboard-dir ${OUTPUT_PATH}/tensorboard
  #--wandb-project ${WANDB_PROJECT}
  #--wandb-exp-name ${GPU}-Qwen3-30B-A3B-TP${TP}PP${PP}EP${EP}CP${CP}VPP${VPP}-MBS${MBS}GBS${GBS}-${COMMENT}

  # Add mixed precision args
  --bf16 
)

# Full recompute
if [[ "${FULL_RECOMPUTE}" == "1" ]]; then
  COMMENT=${COMMENT}_full_recompute
  MEGATRON_ARGS+=(
    --recompute-granularity full
    --recompute-method uniform 
    --recompute-num-layers 1
  )
fi

if [[ "${MOE_GROUPED_GEMM}" == "1" ]]; then
    MEGATRON_ARGS+=("--moe-grouped-gemm")
fi

if [[ "$MOE_ENABLE_DEEPEP" == "1" ]]; then
    MEGATRON_ARGS+=("--moe-enable-deepep")
fi

if [[ -n "$LOAD_PATH" ]]; then
    MEGATRON_ARGS+=("--load ${LOAD_PATH}")
fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo ${PROFILE_CMD} torchrun \
        --nnodes ${WORLD_SIZE:-1} \
        --node_rank ${RANK:-0} \
        --nproc_per_node ${NGPU:-8} \
        --master_addr ${MASTER_ADDR:-"127.0.0.1"} \
        --master_port ${MASTER_PORT:-12345} \
        ${MEGATRON_PATH}/pretrain_gpt.py \
        ${MEGATRON_ARGS[@]}
else
    ${PROFILE_CMD} torchrun \
        --nnodes ${WORLD_SIZE:-1} \
        --node_rank ${RANK:-0} \
        --nproc_per_node ${NGPU:-8} \
        --master_addr ${MASTER_ADDR:-"127.0.0.1"} \
        --master_port ${MASTER_PORT:-12345} \
        ${MEGATRON_PATH}/pretrain_gpt.py \
        ${MEGATRON_ARGS[@]}
fi
