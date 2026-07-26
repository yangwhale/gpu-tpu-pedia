#!/bin/bash
# 在 pod 里跑：curl + bearer token 并行拉 GCS 对象。
#
# 为什么不用 gcloud: vLLM deepgemm 镜像里没有 gcloud / gsutil / wget / aria2c，
#   只有 curl + python3。实测 16 路并行 2.7 GB/s/pod，比 gcloud 还快。
# 为什么不用 kubectl cp: 走 API server，慢一个量级。
#
# 用法（本机侧先备好清单和 token，再 cp 进 pod）:
#   gcloud storage ls -r 'gs://<bucket>/<PREFIX>/**' | sed 's|gs://<bucket>/||' \
#     | grep -v '/$' > /tmp/objs.list
#   gcloud auth application-default print-access-token > /tmp/tok
#   for f in tok objs.list pull-gcs-model.sh; do kubectl cp /tmp/$f <pod>:/tmp/$f; done
#   kubectl exec <pod> -- bash -c "BUCKET=<bucket> PREFIX=<PREFIX> \
#     setsid nohup bash /tmp/pull-gcs-model.sh > /tmp/pull.log 2>&1 </dev/null &"
#
# ⚠️ access token 只有 1 小时有效期。拉 800G+ 的模型前先刷新，
#    否则脚本会「跑完、打印 DONE、但一个 shard 都没下来」。
# ⚠️ 断点续传靠 `[ -s "$f" ]` 跳过已完成的；中断后删掉 *.part 再跑一遍即可补齐。
set -o pipefail
BUCKET=${BUCKET:-chrisya-gb300-models}
PREFIX=${PREFIX:-DeepSeek-V4-Pro-DSpark}
LIST=${LIST:-/tmp/objs.list}
DST=${DST:-/mnt/ssd}
JOBS=${JOBS:-16}
TOK=$(cat "${TOKFILE:-/tmp/tok}")

get(){
  o="$1"; f="$DST/$o"
  mkdir -p "$(dirname "$f")"
  [ -s "$f" ] && return 0
  # 对象名含 / 必须 URL-encode
  enc=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "$o")
  curl -sfL -H "Authorization: Bearer $TOK" \
    "https://storage.googleapis.com/storage/v1/b/$BUCKET/o/$enc?alt=media" \
    -o "$f.part" && mv "$f.part" "$f"
}
export -f get; export TOK BUCKET DST

xargs -a "$LIST" -P "$JOBS" -I{} bash -c 'get "$@"' _ {}

echo "DONE $(du -sh "$DST/$PREFIX" 2>/dev/null | cut -f1) / $(ls "$DST/$PREFIX"/*.safetensors 2>/dev/null | wc -l) shards"
# 校验：实际 shard 数要等于清单里的 safetensors 数。不等 = token 过期或网络中断，重跑即可
echo "期望 $(grep -c '\.safetensors$' "$LIST") shards"
