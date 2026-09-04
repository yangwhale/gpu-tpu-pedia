#!/usr/bin/env bash
#
# 把整条构建链包成一条命令。
#
#     Courses/tools/build-all.sh            # 全量重建 + 版面体检
#     Courses/tools/build-all.sh --lint     # 只跑体检，不重建
#
# ⭐ 为什么要有这个脚本
#
# 这条链有六步、两个必须记住的坑，而它一直靠人脑记着顺序。忘一步不会报错 ——
# 只会让页面上留着上一版的内容，而且看不出来。把顺序写进文件，人就不用记了。
#
# ⛔ 坑一：tpu-micro/build_doc.py 默认是 --mode internal，而输出落在**公开仓库**。
#    忘了传 --mode public，内部内容就会被渲染进去，然后被 commit。
#    （闸门只挡渲染产物，见 memory feedback_gate-filters-output-not-source。）
#
# ⛔ 坑二：它的输出路径是**第一个位置参数**，不是 -o。不传就默默写去 /tmp，
#    构建"成功"了但页面一个字没变。
#
# 这两条都已经写死在下面，不要再手敲这两行命令。
#
# 关于顺序：§0–§2 那十五张图不用单独跑 —— topic02-port-microscope.py 会自己
# 调 topic02-inject-s012.py，后者再去跑那十个画图脚本。所以画图脚本改了，
# 跑这一条就够了。
set -euo pipefail
cd "$(dirname "$0")"

W=../WebPages
step() { printf '\n\033[1m▸ %s\033[0m\n' "$1"; }

if [ "${1:-}" != "--lint" ]; then
  step "GPU 显微镜"
  python3 gpu-micro/build_doc.py $W/gpu-microscope.html

  step "TPU 显微镜（public 模式 —— 见文件头坑一/坑二）"
  python3 tpu-micro/build_doc.py $W/tpu-microscope.html --mode public

  step "专题二 L300 §3–§9（顺带重生成并注入 §0–§2 的十五张图）"
  python3 topic02-port-microscope.py

  step "专题二 L200"
  python3 topic02-build-L200.py

  step "专题二 L200 讲义"
  python3 topic02-build-L200-lecture.py

  # ⭐ 专题八在轮到它之前就开工了 —— 材料是讲专题二时问出来的，
  #    当场写进了它该属于的那一讲。它的 CSS 从专题二 L300 / 专题一讲义抽，
  #    **所以必须排在那两步之后**。
  step "专题八 教材"
  python3 topic08-build.py

  step "专题八 讲义"
  python3 topic08-build-lecture.py
fi

step "版面体检（报告为主，不中止）"
python3 topic02-lint-readability.py

printf '\n\033[1m▸ 产物\033[0m\n'
for f in topic-01.html topic-02-L300.html topic-02.html topic-08.html \
         gpu-microscope.html tpu-microscope.html; do
  [ -f "$W/$f" ] || continue
  printf '  %-24s %9s  %2d 图\n' "$f" \
    "$(wc -c <"$W/$f" | numfmt --to=iec)" "$(grep -c '<figure' "$W/$f" || true)"
done
