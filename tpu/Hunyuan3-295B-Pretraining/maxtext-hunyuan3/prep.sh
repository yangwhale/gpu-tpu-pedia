#!/bin/bash
# 一次性准备：把 hunyuan3 分支的整棵 src/maxtext 打包传到 GCS，供 run.sh 注入容器。
# 只有改了代码才需要重跑；换 XLA flag / 换参数不用。
#
# 用法：GCS_STAGE=gs://your-bucket/hy3 bash prep.sh [branch]
set -euo pipefail
GCS_STAGE=${GCS_STAGE:?需要 GCS_STAGE，例如 gs://my-bucket/hy3}
BRANCH=${1:-hunyuan3}
REPO=${REPO:-https://github.com/yangwhale/maxtext.git}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "[1/3] clone $REPO @ $BRANCH"
git clone -q --depth=1 --single-branch --branch "$BRANCH" "$REPO" "$WORK/mt"
cd "$WORK/mt"
echo "      commit $(git rev-parse --short HEAD)  $(git log -1 --format=%ad --date=short)"

# 自检：确认分支里该有的东西都在。少一样后面到 TPU 上才炸，代价大得多。
echo "[2/3] 自检"
for f in src/maxtext/models/hunyuan3.py \
         src/maxtext/configs/models/hunyuan3-295b.yml \
         src/maxtext/configs/models/hunyuan3-smoke.yml; do
  [ -f "$f" ] || { echo "  ✗ 缺 $f"; exit 1; }
done
grep -q '"hunyuan3-295b"'  src/maxtext/configs/types.py || { echo "  ✗ types.py 白名单缺 hunyuan3-295b"; exit 1; }
grep -q '"hunyuan3-smoke"' src/maxtext/configs/types.py || { echo "  ✗ types.py 白名单缺 hunyuan3-smoke"; exit 1; }
grep -q 'HUNYUAN3' src/maxtext/common/common_types.py || { echo "  ✗ 枚举缺 HUNYUAN3"; exit 1; }
grep -q '_moe_block_attr' src/maxtext/trainers/pre_train/train.py || { echo "  ✗ train.py 的 bias 路径补丁不在"; exit 1; }
# 属性名两边必须一致，否则首步 AttributeError（2026-07-29 踩过）
A=$(grep -c 'Hunyuan3MoeBlock_0' src/maxtext/models/hunyuan3.py)
B=$(grep -c 'Hunyuan3MoeBlock_0' src/maxtext/trainers/pre_train/train.py)
[ "$A" -ge 1 ] && [ "$B" -ge 1 ] || { echo "  ✗ Hunyuan3MoeBlock_0 属性名 model=$A train=$B 对不上"; exit 1; }
echo "      6 项全过"

echo "[3/3] 打包上传"
tar czf "$WORK/hy3-maxtext.tgz" src/maxtext
gsutil -q cp "$WORK/hy3-maxtext.tgz" "$GCS_STAGE/hy3-maxtext.tgz"
echo "      -> $GCS_STAGE/hy3-maxtext.tgz  ($(du -h "$WORK/hy3-maxtext.tgz" | cut -f1))"
