#!/bin/bash
# 在 pod 里跑：用 bearer token + curl 并行拉 GCS 对象（镜像无 gcloud）
TOK=$(cat /tmp/tok); B=chrisya-gb300-models; DST=/mnt/ssd
get(){ o="$1"; f="$DST/$o"; mkdir -p "$(dirname "$f")"
  [ -s "$f" ] && return 0
  curl -sfL -H "Authorization: Bearer $TOK" \
    "https://storage.googleapis.com/storage/v1/b/$B/o/$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "$o")?alt=media" \
    -o "$f.part" && mv "$f.part" "$f"; }
export -f get; export TOK B DST
xargs -a /tmp/dspark.list -P 16 -I{} bash -c 'get "$@"' _ {}
echo "DONE $(du -sh $DST/DeepSeek-V4-Pro-DSpark | cut -f1) $(ls $DST/DeepSeek-V4-Pro-DSpark/*.safetensors 2>/dev/null|wc -l) shards"
