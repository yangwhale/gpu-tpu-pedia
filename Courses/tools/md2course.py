# -*- coding: utf-8 -*-
"""把 `Courses/专题NN-*.md` 转成课程页面的正文 HTML。

════════════════════════════════════════════════════════════════
为什么要有这个东西
════════════════════════════════════════════════════════════════
专题一、二是**先有页面**（手写 / 脚本生成 HTML），大纲 md 只是备忘。
从专题三开始反过来：**大纲 md 已经写到成品质量了**，
再手抄一遍 HTML 就是「同一个职责两个载体」——&nbsp;两边必然走散。

所以规矩改成：**md 是源，页面是产物。** 改内容只改 md。

⛔ 这个转换器**故意做得很小**，只认这门课实际用到的那几种写法：

  `## 一 · 标题`      → 一个 <section>，带 badge
  `### 1.2 标题`      → <h3>
  `#### 标题`         → <h4>
  `| a | b |`         → <table>
  ```` ``` ````       → <pre><code>
  `> …`               → <div class="note">（`> ###` 那行当框题）
  `- ` / `1. `        → <ul> / <ol>
  `**x**` `*x*` `` `x` `` `[t](u)`  → <b> <em> <code> <a>

⚠️ **它不是通用 markdown 渲染器。** 遇到没见过的写法会原样输出，
   不会报错 ——&nbsp;所以**改完 md 一定要看一眼渲染结果**。
   （这条比「让它支持全部语法」重要：全支持意味着这个文件会长成第二个
   markdown 库，而这门课根本用不到那么多。）
"""
import html as _html
import re

# 行内：顺序有讲究 —— 先抠出代码，再处理别的，最后还原
_CODE = []


def _stash(m):
    _CODE.append(m.group(1))
    return "\x00%d\x00" % (len(_CODE) - 1)


# ⛔ md 里是允许直接写 HTML 实体的（这门课到处是 `——&nbsp;`），
#    而 _html.escape 会把它的 & 转成 &amp;，页面上就露出 "&nbsp;" 五个字符。
#    所以转义之前先把实体抠出来，转完再放回去。
_ENT = re.compile(r"&(#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")


def inline(t):
    del _CODE[:]
    t = re.sub(r"`([^`]+)`", _stash, t)
    ents = []
    t = _ENT.sub(lambda m: (ents.append(m.group(0)), "\x01%d\x01" % (len(ents) - 1))[1], t)
    t = _html.escape(t, quote=False)
    t = re.sub(r"\x01(\d+)\x01", lambda m: ents[int(m.group(1))], t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", t)
    t = re.sub(r"~~([^~]+)~~", r"<s>\1</s>", t)
    t = re.sub(r"\x00(\d+)\x00",
               lambda m: "<code>%s</code>" % _html.escape(_CODE[int(m.group(1))]), t)
    return t


def _table(rows):
    out = ["<table>"]
    head, body = rows[0], rows[2:]          # rows[1] 是分隔行
    out.append("<thead><tr>" + "".join(
        "<th>%s</th>" % inline(c) for c in head) + "</tr></thead><tbody>")
    for r in body:
        out.append("<tr>" + "".join("<td>%s</td>" % inline(c) for c in r) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def _split_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _note_class(text):
    """按框里的记号挑配色 —— 跟专题一、二那套 note 类保持一致。"""
    if "⛔" in text:
        return "note danger"
    if "⚠️" in text or "🚧" in text:
        return "note warn"
    if "⭐" in text:
        return "note ok"
    return "note info"


def convert(md, section_badge="第 %s 节"):
    """返回 (html, sections)；sections 是 [(锚点, 中文序号, 标题)] 供侧栏用。"""
    lines = md.split("\n")
    out, sections = [], []
    i, in_sec = 0, False
    n = len(lines)

    def close_sec():
        if in_sec:
            out.append("</div></section>")

    while i < n:
        ln = lines[i]

        # ── 代码块 ──────────────────────────────────────────────
        if ln.startswith("```"):
            j = i + 1
            buf = []
            while j < n and not lines[j].startswith("```"):
                buf.append(lines[j])
                j += 1
            out.append("<pre><code>%s</code></pre>"
                       % _html.escape("\n".join(buf), quote=False))
            i = j + 1
            continue

        # ── 引用块 → note ───────────────────────────────────────
        if ln.startswith(">"):
            j = i
            buf = []
            while j < n and (lines[j].startswith(">") or lines[j].strip() == ""):
                if lines[j].strip() == "":
                    # 空行只有在后面还有 > 时才算块内换行
                    if j + 1 < n and lines[j + 1].startswith(">"):
                        buf.append("")
                        j += 1
                        continue
                    break
                buf.append(re.sub(r"^>\s?", "", lines[j]))
                j += 1
            inner = "\n".join(buf)
            title = ""
            m = re.match(r"^#{2,4}\s+(.+)$", buf[0]) if buf else None
            if m:
                title = inline(m.group(1))
                inner = "\n".join(buf[1:])
            body = convert_body(inner)
            out.append('<div class="%s">%s%s</div>' % (
                _note_class(inner + title),
                '<span class="t">%s</span>' % title if title else "",
                body))
            i = j
            continue

        # ── 表格 ────────────────────────────────────────────────
        if ln.startswith("|"):
            j = i
            rows = []
            while j < n and lines[j].startswith("|"):
                rows.append(_split_row(lines[j]))
                j += 1
            out.append(_table(rows) if len(rows) >= 2 else "")
            i = j
            continue

        # ── 标题 ────────────────────────────────────────────────
        m = re.match(r"^## (.+)$", ln)
        if m:
            close_sec()
            in_sec = True
            t = m.group(1)
            mm = re.match(r"^([零一二三四五六七八九十]+) · (.+)$", t)
            if mm:
                no, title = mm.group(1), mm.group(2)
                anchor = "s" + no
                sections.append((anchor, no, title))
                out.append('<section id="%s"><div class="wrap">'
                           '<div class="stn"><span class="badge">%s</span>'
                           "<h2>%s</h2></div>" % (anchor, section_badge % no, inline(title)))
            else:
                anchor = "x%d" % len(sections)
                sections.append((anchor, "", t))
                out.append('<section id="%s"><div class="wrap">'
                           '<div class="stn"><h2>%s</h2></div>' % (anchor, inline(t)))
            i += 1
            continue
        m = re.match(r"^### (.+)$", ln)
        if m:
            out.append("<h3>%s</h3>" % inline(m.group(1)))
            i += 1
            continue
        m = re.match(r"^#### (.+)$", ln)
        if m:
            out.append("<h4>%s</h4>" % inline(m.group(1)))
            i += 1
            continue
        if ln.startswith("# "):          # 文档大标题由页面自己的 hero 负责
            i += 1
            continue

        # ── 列表 ────────────────────────────────────────────────
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", ln)
        if m:
            ordered = not m.group(2) in ("-", "*")
            tag = "ol" if ordered else "ul"
            j, items = i, []
            while j < n:
                mm = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[j])
                if mm:
                    items.append(mm.group(3))
                    j += 1
                elif lines[j].startswith(("  ", "\t")) and lines[j].strip() and items:
                    items[-1] += " " + lines[j].strip()      # 续行
                    j += 1
                else:
                    break
            out.append("<%s>%s</%s>" % (
                tag, "".join("<li>%s</li>" % inline(x) for x in items), tag))
            i = j
            continue

        # ── 分隔线 / 空行 / 普通段落 ─────────────────────────────
        if ln.strip() == "---":
            out.append("<hr>")
            i += 1
            continue
        if ln.strip() == "":
            i += 1
            continue
        j, buf = i, []
        while j < n and lines[j].strip() and not re.match(
                r"^(#{1,4} |\||>|```|---$|\s*([-*]|\d+\.)\s)", lines[j]):
            buf.append(lines[j].strip())
            j += 1
        out.append("<p>%s</p>" % inline(" ".join(buf)))
        i = j

    close_sec()
    return "\n".join(x for x in out if x), sections


def convert_body(md):
    """引用块内部：不开 section，只要正文。"""
    h, _ = convert(md)
    return re.sub(r"</?section[^>]*>|<div class=\"wrap\">|</div>(?=\s*$)", "", h)
