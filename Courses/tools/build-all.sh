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
  # ⛔⛔ 2026-09-10 更正：这两行原来写着「源是 md，页面由 md2course.py 转出来，
  #    改内容改那份 md」——&nbsp;**全错，而且是会把人带沟里的那种错**。
  #    真实情况（见 topic03-build.py 文件头）：
  #      · **HTML 就是源**，正文直接写在 topic03-build.py 里（107 KB）
  #      · `Courses/专题03-注意力演进.md` 只是**随手记的大纲，不必与页面同步**
  #        （实测已严重落后：md 停在 9/4 的旧结构，页面早就多出 §零 RNN、
  #         §一 MHA、FlashAttention 那七个小节、Mamba、各家速查……）
  #      · `md2course.py` **已删**
  #    ⭐ 判据：**注释指向一个不存在的工具时，它不会报错 ——&nbsp;
  #      只会让下一个人改错文件，然后奇怪为什么页面没变。**
  step "专题三 教材（从 md 生成）"
  # ⛔ 图必须先生成 —— topic03-build.py 会 assert 找不到 fig3-chronicle.svg。
  #    这条依赖是**故意做成硬失败**的：图缺了宁可构建挂掉，也不要悄悄出一份没图的教材。
  # ⛔ 画法基元在 topic03_draw.py，三个 fig 脚本共用一份 —— 别在各自脚本里另起一套。
  python3 topic03-fig-arc.py           # 总纲：这是一个什么故事（六段骨架）
  python3 topic03-fig-rnn.py           # §零 RNN 四张图（怎么算 / 为什么慢 / 解码又变回来 / 痛点通向哪）
  python3 topic03-fig-mha.py           # §一 MHA 四张图（换掉了什么 / 信息怎么流 / 在算什么 / 多头）
  python3 topic03-fig-motives.py       # §二 两条动机线（硬件账 ＋ 信息账 ＋ 交汇处）
  python3 topic03-fig-knobs.py         # §四 三个旋钮：全课骨架（为什么恰好是三个）
  python3 topic03-fig-knob1.py         # §五 旋钮①（四种存法对照 ＋ RoPE 为什么单走一路）
  python3 topic03-fig-mla-why.py       # §五 MLA 凭什么能压（信息账：白送的 vs 赌出来的）
  python3 topic03-fig-mqa-why.py       # §五 砍头这一支怎么想出来的（2019 选项单 ＋ 自由度不是宽度）
  python3 topic03-fig-knob2.py         # §六 旋钮②（五种读法画成五张 mask）
  python3 topic03-fig-dsa-why.py       # §六 DSA 凭什么只看 2048 个（鸡生蛋 ＋ 让真注意力当老师）
  python3 topic03-fig-nsa-why.py       # §六 NSA 三条路被四个坑逼出来（计算稀疏 ≠ 访存稀疏）
  python3 topic03-fig-chronicle.py     # 上半：时间轴 SVG
  # ⭐ 下半那张 39 行模型表现在是**可排序的 HTML 表**，不是 SVG ——
  #   两边读同一份 topic03_models.py，⛔ 数据只有一份。
  python3 topic03-table-models.py
  # ⭐ 贯穿全篇的主线图（一层 Transformer ＋ 四个变体）—— 一个脚本吐五张。
  #    ⛔ 五张必须同源，别拆成五个脚本：这个教学装置的全部价值就在于
  #      「除了高亮那一处，五张完全一样」，拆开一定会漂而且不报错。
  python3 topic03-fig-transformer.py
  python3 topic03-build.py

  step "专题三 讲义"
  python3 topic03-build-lecture.py

  step "专题八 教材"
  python3 topic08-build.py

  step "专题八 讲义"
  python3 topic08-build-lecture.py

  # ⭐ 专题二外传（L100，主线 17 分钟 ＋ §三 可跳 2 分钟）。CSS 从 L300 抽，**必须排在它之后**。
  #   ⛔ 五张图先生成 —— topic02x-build.py 找不到 svg 会直接 assert 挂掉，
  #     这是故意的：宁可构建失败，也不要悄悄出一份缺图的教材。
  #   ⭐ 画法基元共用 topic03_draw.py，别另起一套。
  step "专题二 外传（L100 · v6e 与扩散模型）"
  for g in topic02x-fig-ridge.py topic02x-fig-v6e.py topic02x-fig-h100.py \
           topic02x-fig-load.py topic02x-fig-diff.py topic02x-fig-scale.py \
           topic02x-fig-sparsecore.py topic02x-fig-torus.py \
           topic02x-fig-why.py topic02x-fig-waist.py \
           topic02x-fig-place.py topic02x-fig-timeline.py \
           topic02x-fig-routes.py; do
    python3 "$g"
  done
  python3 topic02x-build.py

  # ⭐ L200 精讲版。⛔ **必须排在 L100 之后** —— 它的写盘自检会去读 topic-02x.html，
  #   核对「§零–§四 两页节号一一对应」这条约定还成不成立。
  step "专题二 外传 L200（精讲版）"
  python3 topic02x-build-L200.py

  step "专题二 外传 讲义"
  python3 topic02x-build-lecture.py
fi

# ⭐⭐ 配色收尾。⛔ **必须排在所有生成器之后、所有体检之前**：
#   在生成器之后，是因为它改的是产物 —— 2026-09-08 我先直接去改
#   topic-02-L300.html 里的一行 CSS，**下一次 build 原样盖回去，而且不报错**。
#   在体检之前，是因为体检该验的是读者真正拿到的那一版。
#   它是幂等的，重跑无副作用。规则和「为什么不去改画图脚本」写在文件头。
step "配色收尾（500 主色文字降 900、大面板去底）"
python3 topic-repalette.py

step "版面体检（报告为主，不中止）"
python3 topic02-lint-readability.py

# ⭐ 这一条是「渲染后量几何」，跟上面那条「扫源码」互补，**两条都要跑**。
# 2026-09-04 的教训：宽图整张跑出屏幕左边 286px，源码扫不出来、
# 构建全绿、连横向溢出探针都查不到（往左跑不产生滚动条）——
# 只有开一个 1900px 的无头浏览器量 getBoundingClientRect 才看得见。
step "版面体检 · 渲染后几何（左跑 / 右撑 / 图内文字撞车）"
python3 topic02-lint-layout.py

# ⭐ 第三条体检：讲义 ↔ 课件对账。前两条都只看**一个**页面自己是不是自洽，
# 而这一条看的是**两个页面之间**：讲义说「滚到 X」，X 在课件里还在吗。
# ⛔ 这类债不报错、不难看，**只在现场翻车** —— 台上照着念、往下滚、框不在了。
step "讲义 ↔ 课件对账（「滚到 X」的 X 还在不在）"
python3 topic02-lint-cues.py

# ⭐ 第四条体检：**同一份文档内部的指路**。跟上一条是一对 ——
# 那条查「讲义指课件」，这条查「课件指自己」：正文写「§X.Y」，那一节存在吗。
# ⛔ 同样是不报错、不难看、只在读者手上翻车的一类债。2026-09-05 首次跑出 12 处。
# 它抓到的最值钱的一条不是编号错，是**没兑现的承诺**：
# 「这根轴第 2.8 节还会回来一次」—— 而这一版根本没有 2.8，那根轴再没回来过。
step "跨节指针体检（「§X.Y」那一节真的存在吗）"
python3 topic02-lint-xref.py

# ⭐ 第五条体检：**讲义覆盖**。跟第三条是一对 ——
# 第三条查「讲义指的那个东西还在不在」，这条查「教材有的东西讲义讲了没有」。
# ⛔ 2026-09-06 一天之内同一个病犯了两次，两次都是**讲到那儿才被现场抓住**：
#    ① §9 的教材精简过，讲义还在讲已经删掉的「八行取舍表」；
#    ② §5.4b（TMA）在教材里是主线，讲义里一个字都没有。
#    第三条 lint 对这两件事完全无感 —— 它查的「滚到 X」的 X 确实都还在。
# ⭐ 形状：**讲义是教材的下游，教材增删它都不会自己跟着动，而两边都不报错。**
step "讲义覆盖体检（教材有的小节，讲义讲了没有）"
python3 topic02-lint-coverage.py

# ⭐ 第六条体检：**<head> 里的元信息**。前五条查的都是页面上看得见的东西，
# 而标题只出现在浏览器标签、og 只出现在分享卡片上 —— ⛔ 这类错能错很久没人发现。
# 2026-09-07 首次跑出：专题三、专题八的标题和四条 og 全是从专题二整段搬过来的，
# 因为那句改标题的 str.replace 模式匹配不上，而**匹配不上是静默的**。
# ⭐ 第七条体检：**节号 ↔ 小节号**。2026-09-08 立的，起因是专题三两次重编号之后
# §五～§八 四节的小节号集体比节号少 1 —— 页面上写着「第 六 节」，小节却编成 5.x。
# ⛔ 已有的五条一条都没抓到：跨节指针体检查的是「§X.Y 指的那节存不存在」，
#    而 5.1 确实存在（只是长错了地方），指向它的 §5.5b 也就顺理成章地通过了。
# ⭐⭐ 形状：**整片一起错位时，任何只做「内部一致性」的检查都会放行。**
#    必须引入外部锚点 —— 这里的锚点是节标题上那个中文数字。
step "节号 ↔ 小节号对账（「第 六 节」里的小节该是 6.x）"
python3 topic02-lint-secnum.py

step "head 元信息体检（标题 / og 指向 / og 图存在）"
python3 topic02-lint-meta.py
# ⛔ 2026-09-10 加：四页 famnav 必须跟 topic02_family.PAGES 一致。
#   L300 那份是手写在 HTML 里的，改过时长后它独自留在旧值上（写着 20 分钟），
#   而 family.py 写 19、实际 17 —— 三个版本，谁都不报错。
python3 topic02-lint-meta.py --famnav

printf '\n\033[1m▸ 产物\033[0m\n'
for f in topic-01.html topic-02-L300.html topic-02.html topic-02x.html \
         topic-02x-L200.html \
         topic-03.html topic-08.html \
         gpu-microscope.html tpu-microscope.html; do
  [ -f "$W/$f" ] || continue
  printf '  %-24s %9s  %2d 图\n' "$f" \
    "$(wc -c <"$W/$f" | numfmt --to=iec)" "$(grep -c '<figure' "$W/$f" || true)"
done
