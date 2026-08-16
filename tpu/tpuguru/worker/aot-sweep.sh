#!/bin/bash
# 并行 AOT 扫描 —— AOT 是纯 CPU 任务，80 核不该排队跑
MODEL=${MODEL:-hunyuan3-295b}; LAYERS=${LAYERS:-0}; CPUS=${CPUS:-11}
# LAYERS=0 表示用模型自带的生产层数
FLAGS='--xla_tpu_dvfs_p_state=7 --xla_tpu_scoped_vmem_limit_kib=65472 --xla_enable_async_all_gather=true --xla_tpu_enable_sparse_core_reduce_scatter_v2=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true --xla_tpu_enable_sparse_core_collective_aggregator=true --xla_tpu_use_tc_device_shape_on_sc=True --xla_sc_disable_megacore_partitioning=True --xla_tpu_enable_latency_hiding_layer_scheduler=true --xla_tpu_scheduler_percent_shared_memory_limit=150 --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true --xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false'
TILE=""; for m in wi wo; do for p in fwd dlhs drhs; do
  TILE="$TILE ${m}_tile_${p}_batch_seq=512 ${m}_tile_${p}_embed_dim=2048 ${m}_tile_${p}_mlp_dim=1536"
done; done
LAYER_ARG=""; [ "$LAYERS" != "0" ] && LAYER_ARG="base_num_decoder_layers=$LAYERS"
one() {
  local P=$1
  docker run --rm --cpus=$CPUS -v $HOME/aot-ablation/work:/w \
    gcr.io/chris-pgp-host/chrisya-maxtext-latest:latest bash -c "
    set -e; cd /deps && rm -rf src/maxtext && tar xzf /w/hy3-maxtext.tgz
    export JAX_PLATFORMS=cpu TF_CPP_MIN_LOG_LEVEL=2
    python3 -m src.maxtext.trainers.pre_train.train_compile src/maxtext/configs/base.yml \
      model_name=$MODEL run_name=a base_output_directory=/tmp/o \
      $LAYER_ARG compile_topology=tpu7x-128 compile_topology_num_slices=1 \
      compile_xla_flags=\"$FLAGS\" dataset_type=synthetic enable_checkpointing=False steps=3 \
      dtype=bfloat16 weight_dtype=float32 override_model_config=True \
      ici_fsdp_parallelism=-1 ici_tensor_parallelism=1 megablox=True sparse_matmul=True scan_layers=True \
      sa_block_q=2048 sa_block_kv=2048 sa_block_kv_compute=2048 sa_block_q_dkv=2048 \
      sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=2048 sa_block_q_dq=2048 sa_block_kv_dq=2048 \
      remat_policy=custom decoder_layer_input=offload attention=flash \
      allow_split_physical_axes=True tokenizer_type=tiktoken \
      tokenizer_path=src/maxtext/assets/tokenizer_llama3.tiktoken \
      per_device_batch_size=$P max_target_length=4096 use_custom_sort_vjp=True \
      sa_use_fused_bwd_kernel=True use_tokamax_splash=True out_proj=remat \
      opt_type=adamw mu_dtype=bfloat16 grad_dtype=bfloat16 use_iota_embed=True \
      use_qwix_quantization=True quantization=fp8_full weight_quantization_calibration_method=absmax \
      $TILE 2>&1 | grep -E 'temp_size_in_bytes|temporaries \(|Error' | tail -3
  " > /tmp/aot-${MODEL}-L${LAYERS}-p${P}.out 2>&1
  echo "done $P"
}
for P in "$@"; do one $P & done
wait
