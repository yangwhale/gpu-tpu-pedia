# -*- coding: utf-8 -*-
"""内部版 / 公开版的单一真相源 —— 一份稿子，两种输出。

**为什么不是复制两份。** 复制出来的两份一定会漂移：某个数字改了内部那份、
忘了改公开那份，而先发出去的偏偏是公开那份。这里改成「一份稿子 + 一道闸门」，
于是有三个硬性保证：

  1. 公开版**不可能**包含内部版没有的事实（它是同一段文字的子集）
  2. 被滤掉了什么，`audit()` 能逐条列出来 —— 过滤是可审计的，不是靠记性
  3. 忘记标注的内容会**默认出现在公开版里**，所以标注规则必须是
     「拿不准就标 INTERNAL」，而不是反过来

用法：

    from gate import I, P, IP, gated, audit, set_mode

    set_mode("public")
    I("这句只有内部版看得到")            # → ""
    P("这句只有公开版看得到")            # → 原文
    IP("内部说法", "公开说法")           # → "公开说法"
    gated("时钟 1.9 GHz", "时钟官方未公开", why="内部设备表")   # 带审计记录

判 INTERNAL 的标准（宁可多标，不可漏标）：
  · 任何来自雇主内部资料、内部设备表、源码分析、trace 分析的数字
  · 时钟频率、片上 SRAM 的**带宽**（容量若在开源代码里则算公开）
  · 内部代号、bug 号、go/ 链接、内部主机名与路径
  · 未公开产品的规格、未公开的性能实测
"""

_MODE = "internal"
_LOG = []          # [(kind, 内部内容摘要, 公开替代摘要, 为什么内部)]


def set_mode(m):
    global _MODE, _LOG
    assert m in ("internal", "public"), m
    _MODE = m
    _LOG = []


def mode():
    return _MODE


def is_public():
    return _MODE == "public"


def _clip(s, n=72):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 1] + "…"


def I(text, why="内部资料"):
    """只在内部版出现。公开版渲染成空串。"""
    _LOG.append(("drop", _clip(text), "", why))
    return "" if is_public() else text


def P(text):
    """只在公开版出现 —— 通常是用来替代被滤掉那段的说明文字。"""
    return text if is_public() else ""


def IP(internal_text, public_text, why="内部资料"):
    """同一件事的两种说法。公开版必须**仍然成立**，只是少了具体数值。"""
    _LOG.append(("swap", _clip(internal_text), _clip(public_text), why))
    return public_text if is_public() else internal_text


def gated(internal_text, public_text, why="内部资料"):
    """IP 的别名，读起来更像「这里有一道闸门」。"""
    return IP(internal_text, public_text, why)


def audit():
    """返回这次构建里所有被闸门处理过的条目 —— 用来核对过滤是否到位。"""
    return list(_LOG)


def audit_report():
    if not _LOG:
        return "（本次构建没有任何内部内容被处理）"
    rows = ["模式：" + _MODE, f"闸门命中 {len(_LOG)} 处：", ""]
    for i, (kind, a, b, why) in enumerate(_LOG, 1):
        tag = "删除" if kind == "drop" else "替换"
        rows.append(f"{i:>3}. [{tag}] ({why})")
        rows.append(f"     内部：{a}")
        if b:
            rows.append(f"     公开：{b}")
    return "\n".join(rows)


# ══════════════════════════════════════════════════════════════════════
# 出厂自检：公开版里绝不允许出现的字样。build 完成后跑一遍。
# 这是第二道防线 —— 万一某处忘了走闸门，这里要拦下来。
# ══════════════════════════════════════════════════════════════════════
import re

_FORBIDDEN = [
    # 位数是 8 不是 6：`topic-01.html` 里有一句 JS 写着 `(b/1048576).toFixed(1)+" MiB"`
    # —— 字节换 MiB。6 位阈值会把它当成 bug 号报出来。真实 bug 号是 9 位。
    # **一个总在误报的闸门，等于没有闸门**，所以宁可收紧。
    (r"\bb/\d{8,}\b",              "Buganizer bug 号"),
    (r"\bgo/[a-z0-9\-/]+",         "内部 go/ 链接"),
    (r"corp\.google\.com",         "内部域名"),
    (r"cc\.higcp\.com",            "内部站点"),
    (r"g3doc",                     "内部文档系统"),
    (r"内部设备表",                 "内部来源署名"),
    (r"内部资料|内部文档|内部源码",   "内部来源署名"),
    (r"/google/(bin|src)/",        "内部路径"),
    # ⛔ 下面这两条是 2026-08-31 补的。原先只有上面那条 `/google/(bin|src)/`，
    # 而真正漏出去的那一条长的是 `/google_src/files/…` —— **中间是下划线不是斜杠**，
    # 一条都没匹配上，在公开仓库里躺了很久。
    # 教训跟 marker 白名单同类：**规则写得像对的，不等于它覆盖了真实的写法。**
    (r"/google_src/",              "内部路径（下划线写法）"),
    (r"\bgooglefile:",             "内部文件系统前缀"),
]


# 这几类**只在「生成出来的公开页面」上算问题**，在生成器源码里是必要的：
# `build_doc.py` 得能说出「这个数来自内部资料，公开版要换掉」，
# 那句话本身不是泄漏。仓库级闸门（scripts/public-repo-guard.py）按这个集合放行。
PAGE_ONLY = {"内部来源署名"}


def lint_public(html):
    """返回 [(命中的字样, 说明), ...]；空列表表示通过。"""
    hits = []
    for pat, desc in _FORBIDDEN:
        for m in re.finditer(pat, html):
            hits.append((m.group(0), desc))
    # 去重但保留顺序
    seen, out = set(), []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out
