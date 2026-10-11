#!/usr/bin/env bash
# =============================================================================
# 06a-aot-dsv3.sh —— 06b §5：在 CPU 上 AOT（提前编译）DeepSeek-V3 配方，不占一张 TPU 就知道 HBM 装不装得下
#
# 做什么：用配方同一个镜像、同一份参数（06a-dsv3-recipe.env），跑 MaxText 的 train_compile，
#   按目标拓扑（默认 SLICES × tpu7x-512，SLICES 默认 8）把训练步完整编一遍，打印每个 device 的显存账。
#   AOT 报的显存与真机 JIT 的编译报告一致，但都是编译期估计，
#   和实机 HBM 峰值能差几到十几 GiB、两个方向都有（06b §5）。编译期初筛看 temp_size，
#   跟 94.74 GiB/device 比（v7 每芯片 192 GiB ÷ 2 device），不要把 argument 加进去；OOM 最终以实机 peakBytesInUse 为准。
#
# 用法：
#   SLICES=N runbook/scripts/06a-aot-dsv3.sh <名字> [额外 MaxText 参数 …]
#   SLICES=1  runbook/scripts/06a-aot-dsv3.sh s1                    # 单片对照（=实测那一片）
#   runbook/scripts/06a-aot-dsv3.sh smoke base_num_decoder_layers=5  # 先用少层数试通链路（显存不作数）
# 变量：SLICES=N  CPUS=min(64, nproc)（实测大部分时间只用 1 核，偶尔十几核）  OUT=~/runbook-state/aot  IMAGE（默认读 recipe.env 的 digest）
#       TOPO_NAME=tpu7x-512  目标拓扑，按 device 数写（1 cube 4x4x4 = tpu7x-128）；JobSet 换 TOPO 时这里要同步换
#       STEPS=50
#       XLA_FLAGS_ENV=  传进容器的 XLA_FLAGS 环境变量（= JobSet 里的 XLA_FLAGS env）；默认读 recipe.env 的 RECIPE_XLA_FLAGS_ENV
#       EXTRA_XLA=      追加到 compile_xla_flags（= JobSet 里的 EXTRA_LIBTPU_ARGS）；做单 flag A/B 用
# 前提：本机有 docker（当前用户能跑 docker）、磁盘 ≥ 25 GB；镜像**不能匿名拉**：先 `gcloud auth configure-docker us-docker.pkg.dev`
#       （实测匿名取 token 返回 DENIED）。内存：2 片 / 1 cube 并行两个编译时实测峰值见 06b §5；不需要 GCS、TPU、集群凭据。
# 产物：$OUT/aot-<名字>.log；脚本最后打印 temp / argument / host temp（GiB），temp > 94.74 退出码 3。
#   日志里有 'Finished train_compile.py successfully!' 不等于装得下（06b §5 附录坑 2）——以脚本打印的 temp 为准。
# 拓扑名按 device 数：4x8x8 = 256 芯 = 512 device = tpu7x-512（写成 tpu7x-256 编的是 4x4x8，数字全错也不报错）。
# =============================================================================
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; source "$here/06a-dsv3-recipe.env"
NAME=${1:?用法: 06a-aot-dsv3.sh <名字> [额外参数]}; shift || true
N=$(nproc); SLICES=${SLICES:-8}; CPUS=${CPUS:-$(( N < 64 ? N : 64 ))}; OUT=${OUT:-$HOME/runbook-state/aot}; IMAGE=${IMAGE:-$RECIPE_IMAGE}   # 默认 8 只是演示值，不是本书实测的片数
TOPO_NAME=${TOPO_NAME:-tpu7x-512}; STEPS=${STEPS:-50}
XLA_FLAGS_ENV=${XLA_FLAGS_ENV-${RECIPE_XLA_FLAGS_ENV:-}}; EXTRA_XLA=${EXTRA_XLA:-}
mkdir -p "$OUT"; LOG="$OUT/aot-$NAME.log"
echo "[aot $NAME] $TOPO_NAME × $SLICES 片，${CPUS} 核，镜像 ${IMAGE##*@}；XLA_FLAGS='$XLA_FLAGS_ENV'；追加编译 flag '$EXTRA_XLA'；日志 $LOG"
t0=$(date +%s)
set +e
docker run --rm --cpus="$CPUS" -w /deps/src \
  -e JAX_PLATFORMS=cpu -e TF_CPP_MIN_LOG_LEVEL=1 -e PYTHONUNBUFFERED=1 -e XLA_FLAGS="$XLA_FLAGS_ENV" \
  "$IMAGE" python3 -m maxtext.trainers.pre_train.train_compile maxtext/configs/base.yml \
    $RECIPE_MODEL_ARGS \
    dcn_data_parallelism="$SLICES" steps="$STEPS" \
    run_name="aot-$NAME" base_output_directory=/tmp/out \
    compile_topology="$TOPO_NAME" compile_topology_num_slices="$SLICES" \
    compile_xla_flags="$RECIPE_XLA_FLAGS${EXTRA_XLA:+ $EXTRA_XLA}" \
    "$@" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "@@@ rc=$rc wall=$(( $(date +%s) - t0 ))s" | tee -a "$LOG"
# 读数：成功时 train_compile 只打 `Memory analysis: CompiledMemoryStats(... temp_size_in_bytes=...)`，
#   不打 MaxText 的「Total estimated memory size」（核对 14 份日志：成功的 0 份有那一行）；OOM 时打 HLO temporaries。
grep -E "HLO temporaries|Finished train_compile" "$LOG" | sed 's/^/  /' || true
python3 - "$LOG" <<'PY' || rc2=$?
import re, sys
t = open(sys.argv[1], errors="replace").read()
m = re.search(r"CompiledMemoryStats\((.*?)\)", t)
if not m:
    sys.exit(0)                      # 编译没走完（OOM 或配置错）：上面的 HLO temporaries / rc 已说明
kv = dict(re.findall(r"(\w+)_in_bytes=(\d+)", m.group(1)))
g = lambda k: int(kv.get(k, 0)) / 2**30
temp, arg, host = g("temp_size"), g("argument_size"), g("host_temp_size")
print(f"  每 device：temp {temp:.2f} GiB（判据，上限 94.74）  argument {arg:.2f}  host temp {host:.2f}  temp+argument {temp+arg:.2f}")
if temp > 94.74:
    print("  ✗ temp 超过 94.74 GiB：编译「成功」也装不下（06b §5 坑 2）"); sys.exit(3)
print(f"  ✓ 装得下（余 {94.74-temp:.2f} GiB）")
PY
[[ "$rc" == 0 ]] && exit "${rc2:-0}"
exit "$rc"
