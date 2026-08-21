#!/bin/bash
# 批量渲染 3b1b 场景最后一帧，记录成功/失败
OUT=~/3b1b-render/frames; mkdir -p "$OUT"
cd ~/3b1b-videos || exit 1
declare -a JOBS=(
  "mlp.py AlmostOrthogonal"
  "mlp.py ShowAngleRange"
  "mlp.py Superposition"
  "mlp.py MLPIcon"
  "mlp.py LastTwoChapters"
  "mlp.py BasicMLPWalkThrough"
  "mlp.py BreakDownThreeSteps"
  "mlp.py StackOfVectors"
  "mlp.py ClassicNeuralNetworksPicture"
  "mlp.py NonlinearityOfLanguage"
)
for j in "${JOBS[@]}"; do
  set -- $j; f=$1; s=$2
  st=$(date +%s)
  if timeout 300 xvfb-run -a -s "-screen 0 1920x1080x24" \
       /tmp/manim-venv/bin/manimgl "_2024/transformers/$f" "$s" -w -s --hd \
       --video_dir "$OUT" > "/tmp/r-$s.log" 2>&1; then
    echo "OK   $((`date +%s`-st))s  $f  $s"
  else
    echo "FAIL $((`date +%s`-st))s  $f  $s  :: $(grep -oE '[A-Za-z]*Error: .*|Exception: .*' /tmp/r-$s.log | head -1)"
  fi
done
echo "=== 产物 ==="; ls -la "$OUT" 2>/dev/null | tail -15
