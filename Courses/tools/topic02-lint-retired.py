# -*- coding: utf-8 -*-
r"""退役说法体检 ——「这句话已经被判过假了，不许再出现」

⭐⭐ 为什么要有这条

2026-09-14 R48 抓到 fig3-hybrid 的落点带在说一句假话：
「两端各 1/8 的区间里，一个模型都没有」——&#160;而同一格的点阵里，
左端 1:1 上站着两个点、右端 7:1 上站着三个。**图自己在打自己的脸。**

改的时候改了图、改了讲稿，**漏了图注** ——&#160;同一句话住在三个地方：

    画图脚本（图里印的字）   ·   topic03-build.py（图注）   ·   讲义 cue（台上念的）

⛔ 已有的七条体检一条都管不了这个：它们查的是**结构**（小节存不存在、
   cue 指的框还在不在、节号对不对得上），而这是**内容**。
   一句假话在三个地方语法全对、渲染全对、指针全对。

⭐ 形状：**删掉一句错话不是一次修改，是一次「全树清除」。**
   而人只会改自己当时正看着的那一处。

## 这条 lint 怎么工作

下面 RETIRED 里每一条记的是**一句已经判过假的话**：它的正则、为什么假、
什么时候判的、该换成什么。跑的时候扫两处：

  · `../WebPages/*.html` ——&#160;**读者真正拿到的那一版**（图是内联 SVG，所以图里的字也在内）
  · `*.py` 源码 ——&#160;命中时能直接给出该去哪一行改

## ⛔ 最要紧的一条设计：护栏自己要先被验一遍

`RETIRED` 里每条都带一个 `sample` ——&#160;当初那句原话。
启动时先拿正则去匹配它自己的 sample，**匹配不上就直接报错退出**。

为什么必须有这一步：正则写错（少个空格、全角半角、`/` 写成 `／`）时，
这条 lint 会**永远绿**，而绿的原因是它什么都匹配不到。
**一个坏掉的验证器比没有验证器更危险** ——&#160;没有验证器你还知道自己没查，
坏掉的验证器每次都告诉你「查过了，没问题」。

（同型教训：memory `feedback_mutation-test-not-green-check`、
  browser-cli skill 里那次「验证器看不见连发的第 2..N 条，于是自动重发了 7 遍」。）

## 怎么加一条

判掉一句话之后，**在这里补一条**，`sample` 就填当初那句原话。
不要只在 commit message 里写「已删除」——&#160;删除不会阻止它回来。
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..", "WebPages")

# ── 已判假、不许再出现的说法 ────────────────────────────────────────
# phrase : 正则（用 \s* 兜住可能的换行和全角空格）
# sample : ⛔ 当初那句原话。护栏靠它自检，**不能省，也不能随便改**
RETIRED = [
    {
        # ⭐ 2026-09-25 专题五对照构建手册时发现：R7（09-24）就撤回了这句，课件改了口，
        #   讲义第八节收尾那段却原样留着 —— 而那是全讲最后一段，台下记得最牢的位置。
        "id": "t05-一刀逼出一刀",
        "phrase": r"一刀逼出一刀|V3\s*不用\s*TP\s*[，,]?\s*逼出",
        "sample": "这五刀是一刀逼出一刀：batch 小逼出 TP，V3 不用 TP 逼出 EP，上下文一长逼出切序列",
        "why": "只在 FSDP→TP、切序列→PD 分离两处成立；V3 不用 TP 是选型不是被逼，长上下文是模型换了形状。",
        "since": "2026-09-24 R7",
        "instead": "每一刀都在补前面没管到的那一块：batch 小了补上 TP，参数都在专家里补上 EP……",
        "seen_in": "讲义第八节收尾（已改）/ 课件主线（R7 已改）",
    },
    {
        "id": "hybrid-两端各1/8",
        "phrase": r"两端各\s*(1/8|八分之一)[^。；]{0,24}?(一个|一家|一款)[^。；]{0,8}?[都也]?没有",
        "sample": "两端各 1/8 的区间里，一个模型都没有 —— 没有人只掺一两层，"
                  "也没有人敢全用便宜的。",
        "why": "假的。fig3-hybrid 面板②那根轴是 1:1 → 8:1，"
               "左端 1:1 上站着两个点（Gemma 2、gpt-oss-120b），"
               "右端最近的 7:1 上站着三个。两头都不空。",
        "since": "2026-09-14 R48",
        "instead": "最高的一摞在 3:1（十四家占五家）；往右没有一家超过 7:1。"
                   "最左 1:1 那两个清一色是滑窗族，线性混合最省也从 3:1 起步。",
        "seen_in": "画图脚本 / topic03-build.py 图注 / 讲义 §八 cue —— 三处都写过",
    },
    {
        # ⭐⭐ 这条是最好的例子：**图自己 2026-09-08 就判掉了它，而且在代码注释里
        #   写清了为什么**，可图注和讲稿一直照旧印到 09-14 —— 整整活了六天。
        #   ⛔ 「在改动处留一段注释说明为什么改」**不构成清除**。
        #     注释只有改那一处的人会看见，另外两处的人从来不路过。
        "id": "qkv-唯一随长度平方长大",
        "phrase": r"唯一[^。；]{0,6}随长度平方(长大|变大|增长)",
        "sample": "只让他们盯住第一步的形状 —— n 乘 n，整层里唯一一个"
                  "随长度平方长大的东西。",
        "why": "口径错。这说的是**存储**，而 Q·Kᵀ 那一步在 FlashAttention 之后"
               "根本不落地，只在片上过。真正随长度平方长大的是**要算的次数**，"
               "不是显存占用。",
        "since": "2026-09-08 在 fig3-mha-qkv 里改掉，R52 才发现图注和讲稿还留着",
        "instead": "它贵在要算的次数，不在显存。",
        "seen_in": "画图脚本（已改）/ topic03-build.py 图注 / 讲义 §1.2 ——"
                   "只改了第一处，另外两处活了六天",
    },
]

# ── 护栏自检：正则必须能咬住自己的 sample ──────────────────────────
broken = []
for r in RETIRED:
    if not re.search(r["phrase"], r["sample"]):
        broken.append(r["id"])
if broken:
    print("⛔⛔ 退役说法体检**自己坏了**：下面这些正则连自己的 sample 都匹配不上，")
    print("    也就是说它们在这条链上是**永远绿**的 ——&#160;绿的原因是什么都没查。")
    for b in broken:
        print("    · %s" % b)
    print("    ⚠️ 先把正则修好再说。坏掉的验证器比没有验证器更危险。")
    sys.exit(1)


def scan(path, label):
    """返回 [(规则, 行号, 整行)]。二进制/读不了的文件直接跳过。

    ⚠️ `.py` 里以 `#` 开头的行**放行** ——&#160;判掉一句假话之后，
    正确的做法本来就是在原地留一段注释说清「这句为什么假、换成了什么」。
    连那段注释一起告警，就等于在惩罚留档，下次谁也不敢写了。
    ⛔ 但只放行**整行注释**：行尾 `# ...` 不放行，
      因为一行里前半截完全可能是真的在印那句话。

    ⭐⭐ 另一类必须放行的是**引用**：讲稿里有时要写「⛔ 别说成 X」——&#160;
    那是在**禁止** X，不是在**用** X，而且这种句子对讲课的人最有用。
    正则分不清引用和使用（它俩长得一模一样），所以这里**不做启发式猜测**，
    只认一个显式标记 `RETIRED-OK`：

      · `.py` / `.sh` 行尾写 `# RETIRED-OK`
      · HTML 模板行尾写 `<!-- RETIRED-OK -->`（读者看不见，lint 看得见）

    ⛔ 标记必须**跟那句话同一行**。跨行的标记等于给整段开后门。
    ⭐ 显式标记比聪明的正则好，理由只有一条：**加标记是一个动作**。
      写的人必须停下来想一次「我这是在引用还是在用」——&#160;
      而一条能自动识别引用的正则，会在某天悄悄放过一次真正的复活。
    """
    out = []
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().split("\n")
    except (OSError, UnicodeDecodeError):
        return out
    py = path.endswith(".py")
    for r in RETIRED:
        rx = re.compile(r["phrase"])
        for i, ln in enumerate(lines, 1):
            if py and ln.lstrip().startswith("#"):
                continue
            if "RETIRED-OK" in ln:
                continue
            if rx.search(ln):
                out.append((r, i, ln.strip()[:150]))
    return out


targets = ([(p, "产物") for p in sorted(glob.glob(os.path.join(WEB, "*.html")))]
           + [(p, "源码") for p in sorted(glob.glob(os.path.join(HERE, "*.py")))])

hits = []
for path, label in targets:
    # ⚠️ 跳过自己 ——&#160;RETIRED 里存着 sample 原话，不然这条 lint 必然自我命中
    if os.path.abspath(path) == os.path.abspath(__file__):
        continue
    for r, i, ln in scan(path, label):
        hits.append((label, os.path.basename(path), i, r, ln))

if not hits:
    print("   ✅ %d 条已退役的说法，没有一条复活。" % len(RETIRED))
else:
    by_rule = {}
    for label, fn, i, r, ln in hits:
        by_rule.setdefault(r["id"], []).append((label, fn, i, r, ln))
    print("⛔ %d 条退役说法又出现了：" % len(by_rule))
    for rid, rows in by_rule.items():
        r = rows[0][3]
        print("\n══ %s（%s 判掉）" % (rid, r["since"]))
        print("   为什么假：%s" % r["why"])
        print("   该写成　：%s" % r["instead"])
        print("   当初分布：%s" % r["seen_in"])
        for label, fn, i, _r, ln in rows:
            print("   · [%s] %s:%d  %s" % (label, fn, i, ln))
    print("\n   ⚠️ 产物命中 ＝ 读者现在正在读这句话；源码命中 ＝ 下次 build 还会印出来。"
          "\n   ⭐ 一句话住在图、图注、讲稿三个地方 ——&#160;改一处不叫改完。")
