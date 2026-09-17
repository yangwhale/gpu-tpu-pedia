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
  # ⭐ 2026-09-14 R62：名字改成「两版」——&nbsp;原来叫「从 md 生成」，
  #   而 md2course.py 上面那段注释里刚说完它**已删**。
  #   ⛔ 一个步骤名说着一条不存在的流水线，比没有名字更坏。
  step "专题三 教材（主线 L200 ＋ 档案 L300）"
  # ⛔ 图必须先生成 —— topic03-build.py 会 assert 找不到 fig3-chronicle.svg。
  #    这条依赖是**故意做成硬失败**的：图缺了宁可构建挂掉，也不要悄悄出一份没图的教材。
  # ⛔ 画法基元在 topic03_draw.py，三个 fig 脚本共用一份 —— 别在各自脚本里另起一套。
  # ══════════════════════════════════════════════════════════════
  # ⛔⛔ 2026-09-14：这里原来是 **33 行手写的 python3 调用**，而 tools/ 下
  #   已经有 45 个 topic03-fig-*.py ——&nbsp;**12 个从来没被登记过**。
  #   它们的 svg 是早先手跑时落在 tools/ 里的**陈旧产物**，于是：
  #     · 本机构建照样通过（文件在，assert 满足），页面上却是上一版的图；
  #     · 干净 clone 上会直接挂（svg 不在仓库里，gitignore 掉了）。
  #   现场原话「为什么没有清理干净？还有这种该折叠起来的小字？」——
  #   问的就是这 12 张：src() 改了，它们没重跑。
  # ⭐⭐ 判据：**「一份要手工维护的清单」＋「漏了不报错」＝ 一定会漂。**
  #   这跟版面体检漏掉 topic-03、图注还在讲被拆走的半张图，是同一个病。
  #   ⭐ 改法不是「把 12 个补上」（下次加第 46 个还会漏），是**让清单消失**。
  # ⛔ 先删干净再重建：留着旧 svg 的话，某个脚本挂掉时构建会拿上一版顶上，
  #   **静默出一份新旧混合的教材**。宁可 assert 挂掉。
  #   （每张图具体画什么，看各自脚本的 docstring —— 那份不会漂。）
  # ══════════════════════════════════════════════════════════════
  rm -f fig3-*.svg fig3-*.src.html
  for s in topic03-fig-*.py; do python3 "$s"; done
  # ⭐ 39 行模型表是**可排序的 HTML 表**不是 SVG，所以不在上面那个 glob 里。
  #   ⛔ 它跟时间轴图读同一份 topic03_models.py —— 数据只有一份。
  python3 topic03-table-models.py

  # ⭐⭐ 2026-09-14 R62 一分为二：原来的专题三整体降格成 **L300（档案版）**，
  #   旁边新起 **主线 L200**（一条故事线，产物就叫 topic-03.html）。
  #   ⭐ 脚手架（head / 图装配 / 锚点 / 吸顶目录）只有一份：topic03_page.py ——
  #     ⛔ 别让两个生成器各抄一套，那是这个仓库栽过四次的那个形状。
  python3 topic03-build-L300.py
  python3 topic03-build-L200.py

  # ⭐ 两份讲义：主线那份配 topic-03.html，档案那份配 topic-03-L300.html。
  #   ⛔ 新增讲义必须同时在 topic02-lint-cues.py 的 PAIRS 里登记 ——&nbsp;
  #     没登记的会被那条体检主动点名（2026-09-14 就是这么发现漏网的）。
  step "专题三 讲义（主线 ＋ L300）"
  python3 topic03-build-lecture.py
  python3 topic03-build-L300-lecture.py

# ── 专题四 · 反向与优化器 ──────────────────────────────────────────
#   ⭐ 脚手架复用 topic03_page.py，画法基元复用 topic03_draw.py ——
#     ⛔ 别另起一套（专题二 / 三 / 二x 已经共用同一份）。
#   ⚠️ 目前是**骨架**：五个内容节都还挂着「🚧 本章待写」块。
#     写完一章就把 topic04-build.py 里 PLAN 的那一行连同正文占位符一起删。
echo ""
echo "▸ 专题四 教材"
# ⛔ 图先生成 —— topic04-build.py 的 place_figs 找不到 svg 会直接 assert 挂掉。
for s in topic04-fig-*.py; do python3 "$s"; done
python3 topic04-build.py

echo ""
echo "▸ 专题四 讲义"
# ⛔ 必须排在教材之后 —— 讲义的 board cue 要跟成品 topic-04.html 对账。
python3 topic04-build-lecture.py

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

# ⭐ 投屏体检。⛔ 2026-09-13 二轮学生量出来的：整张图投到 1280×720 上，
#   34 张里没有一张能保住 10px。⭐ 根因不是字号小，是**一张三栏图本来就是三页内容**
#   —— 所以这条 lint 给的是一个**读数**（「这张整张投出去最小字多少」），
#   不是一个「把字改大」的指令。低于 9px 就意味着这张图必须一次放大一格讲。
step "投屏体检（整张投到 1280×720 时的最小字号）"
python3 topic02-lint-projection.py

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
python3 topic02-lint-figpos.py

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

# ⭐ 第八条体检：**退役说法**。前七条查的全是**结构** —— 小节存不存在、
# cue 指的框还在不在、节号对不对得上。而这一条查的是**内容**：
# 一句已经被判过假的话，有没有在别处活着。
# ⛔ 2026-09-14 R48 的起因：fig3-hybrid 的落点带写「两端各 1/8 的区间里，
#    一个模型都没有」，而同一格的点阵左端站着两个点 —— **图在打自己的脸**。
#    改的时候改了图、改了讲稿，**漏了图注**，七条体检没有一条有感觉：
#    那句话语法全对、渲染全对、指针全对，它只是不真。
# ⭐⭐ 形状：**一句话住在三个地方（图 / 图注 / 讲稿），删一处不叫删掉。**
step "退役说法体检（判过假的话，有没有在别处活着）"
python3 topic02-lint-retired.py

step "head 元信息体检（标题 / og 指向 / og 图存在）"
python3 topic02-lint-meta.py
# ⛔ 2026-09-10 加：四页 famnav 必须跟 topic02_family.PAGES 一致。
#   L300 那份是手写在 HTML 里的，改过时长后它独自留在旧值上（写着 20 分钟），
#   而 family.py 写 19、实际 17 —— 三个版本，谁都不报错。
python3 topic02-lint-meta.py --famnav

printf '\n\033[1m▸ 产物\033[0m\n'
# ⛔⛔ 2026-09-15 R11：全仓扫一遍「内联 SVG 里混进 HTML 专属标签」。
#   这些页面的 SVG 是**内联**的，HTML 解析器在 foreign content 里碰到
#   <u> <b> <br> 这类只属于 HTML 的元素会**当场退出 foreign content** ——
#   等于就地补了个 </svg>，那之后的图内容整段掉到图外面。
#   ⭐ 实际case：fig3-gun 的收尾落点带只显示两行，剩下四行跑到图下面。
#     而 SVG 源码里那六行一行不少、y 坐标全在带子里 —— 读源码查不出来。
#   📌 topic03 那一家在 topic03_page.finish 里已经会当场中止；
#     这里是**给其余各页兜底**（它们走别的脚手架）。
python3 - "$W" <<'PY'
import re, sys, glob, os
bad = 0
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.html"))):
    s = open(p, encoding="utf-8").read()
    for m in re.finditer(r"<svg\b.*?</svg>", s, re.S):
        seg = re.sub(r"<foreignObject\b.*?</foreignObject>", "",
                     m.group(0), flags=re.S)   # 里面的 HTML 合法
        for t in re.finditer(r"<(u|b|i|em|strong|br|p|div|span|small|code)\b",
                             seg, re.I):
            bad += 1
            print("    ⛔⛔ %s 的内联 SVG 里有 HTML 标签 <%s>"
                  % (os.path.basename(p), t.group(1)))
if bad:
    sys.exit("内联 SVG 里不能有 HTML 标签 —— 它会把 SVG 就地截断。"
             '要下划线用 tspan text-decoration="underline"')
print("   ✅ 所有页面的内联 SVG 里没有 HTML 专属标签")
PY

for f in topic-01.html topic-02-L300.html topic-02.html topic-02x.html \
         topic-02x-L200.html \
         topic-03.html topic-03-L300.html topic-08.html \
         gpu-microscope.html tpu-microscope.html; do
  [ -f "$W/$f" ] || continue
  printf '  %-24s %9s  %2d 图\n' "$f" \
    "$(wc -c <"$W/$f" | numfmt --to=iec)" "$(grep -c '<figure' "$W/$f" || true)"
done
