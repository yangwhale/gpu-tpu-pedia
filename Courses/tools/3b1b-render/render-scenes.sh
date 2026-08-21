#!/usr/bin/env bash
# 批量渲染第一课要用的 3b1b 场景（每个取最后一帧），逐条打印成功/失败与耗时。
#
# 前置：先跑 ./setup-render-env.sh
# 用法：./render-scenes.sh          渲全部
#       ./render-scenes.sh mlp      只渲名字含 mlp 的
set -uo pipefail

VENV="${VENV:-$HOME/manim-venv}"
REPO="${REPO:-$HOME/3b1b-videos}"
OUT="${OUT:-$HOME/3b1b-render/frames}"
LOGS="${LOGS:-$HOME/3b1b-render/logs}"
FILTER="${1:-}"

mkdir -p "$OUT" "$LOGS"
cd "$REPO" || { echo "找不到 $REPO，先跑 setup-render-env.sh"; exit 1; }

# 第一课五节要用的场景，顺序按节。每张图讲什么见 教学材料/README.md
JOBS=(
  # 1 · 会接话的机器
  "auto_regression.py SimpleAutogregression"
  "auto_regression.py AnnotateNextWord"
  "auto_regression.py AthleteCompletion"
  "ml_basics.py TweakedMachine"
  "ml_basics.py DistinguishWeightsAndData"
  # 2 · 词变成向量
  "embedding.py DiscussTokenization"
  "embedding.py IntroduceEmbeddingMatrix"
  "embedding.py ThreeDSpaceExample"
  "embedding.py ManyIdeasManyDirections"
  "embedding.py DotProducts"
  # 3 · Attention
  "attention.py AttentionPatterns"
  "attention.py QueryMap"
  "attention.py KeyMap"
  "attention.py ShowMasking"
  "attention.py DescribeAttentionEquation"
  # 4 · MLP
  "mlp.py MLPIcon"
  "mlp.py BreakDownThreeSteps"
  "mlp.py BasicMLPWalkThrough"
  "mlp.py NonlinearityOfLanguage"
  "mlp.py ShowAngleRange"
  "mlp.py Superposition"
  "mlp.py AlmostOrthogonal"
  # 5 · 参数账
  "attention.py CountMatrixParameters"
  "ml_basics.py ShowGPT3Numbers"
  "ml_basics.py SoftmaxBreakdown"
  # 备用
  "mlp.py StackOfVectors"
  "mlp.py ClassicNeuralNetworksPicture"
  "mlp.py LastTwoChapters"
)

ok=0; fail=0
for j in "${JOBS[@]}"; do
  # shellcheck disable=SC2086
  set -- $j; f=$1; s=$2
  if [ -n "$FILTER" ] && [[ "$f$s" != *"$FILTER"* ]]; then continue; fi
  st=$(date +%s)
  if timeout 300 xvfb-run -a -s "-screen 0 1920x1080x24" \
       "$VENV/bin/manimgl" "_2024/transformers/$f" "$s" -w -s --hd \
       --video_dir "$OUT" > "$LOGS/$s.log" 2>&1; then
    printf 'OK   %3ds  %-16s %s\n' "$(( $(date +%s) - st ))" "${f%.py}" "$s"
    ok=$((ok + 1))
  else
    printf 'FAIL %3ds  %-16s %s :: %s\n' "$(( $(date +%s) - st ))" "${f%.py}" "$s" \
      "$(grep -oE '[A-Za-z]+Error:.*|Exception:.*' "$LOGS/$s.log" | tail -1 | cut -c1-80)"
    fail=$((fail + 1))
  fi
done

echo
echo "成功 $ok / 失败 $fail　　图在 $OUT，日志在 $LOGS"
echo "累计 PNG：$(find "$OUT" -name '*.png' | wc -l)"

# 已知渲不出来的两个（原因清楚，别再浪费时间查）：
#   attention/IntroduceValueMatrix
#     上游场景与当前 manimgl 不兼容 —— FadeTransform 里往 VGroup 塞 ImageMobject。
#   embedding/KingQueenExample
#     要 gensim.downloader 拉 1.6 GB 词向量模型，成本不值。
#     替代：ThreeDSpaceExample / ManyIdeasManyDirections 表达的是同一个意思。
