#!/bin/bash
# vLLM decode (TP4, kv_consumer, FULL_DECODE_ONLY cudagraph)
# 用法: bash vllm-decode-tp4.sh <本 pod IP>
# ⚠️ 必须等 prefill 就绪(curl :8001/health = 200)后再起本进程
# ⚠️ 参数是【本 pod 自己的 IP】—— VLLM_NIXL_SIDE_CHANNEL_HOST 是「本进程 bind 哪个地址」，
#    填成 prefill 的 IP 会 zmq.error.ZMQError: Cannot assign requested address
SELF_IP=$1
export UCX_TLS=cuda_copy,cuda_ipc,tcp
export UCX_CUDA_IPC_ENABLE_MNNVL=y
export UCX_NET_DEVICES=all
export VLLM_USE_NCCL_SYMM_MEM=0 NCCL_CUMEM_ENABLE=1 NCCL_MNNVL_ENABLE=1 NCCL_NVLS_ENABLE=1
export VLLM_NIXL_SIDE_CHANNEL_PORT=5558 VLLM_NIXL_SIDE_CHANNEL_HOST=$SELF_IP
export PYTHONHASHSEED=0
export TMPDIR=/mnt/ssd/tmp && mkdir -p $TMPDIR
vllm serve /mnt/ssd/DeepSeek-V4-Pro-DSpark --served-model-name deepseek-ai/DeepSeek-V4-Pro-DSpark \
  --trust-remote-code --enable-cumem-allocator --kv-cache-dtype fp8 --block-size 256 \
  --port 8002 --tensor-parallel-size 4 \
  --max-num-seqs 1024 --max-num-batched-tokens 8192 --max-cudagraph-capture-size 1024 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[8,16,24,32,40,48,56,64,96,128,192,256,384,512,768,1024]}' \
  --gpu-memory-utilization 0.9 --no-disable-hybrid-kv-cache-manager \
  --attention_config.use_fp4_indexer_cache=True \
  --moe-backend deep_gemm_mega_moe --enable-expert-parallel --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}' \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
