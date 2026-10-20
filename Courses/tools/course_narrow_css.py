# -*- coding: utf-8 -*-
r"""课件页的**窄屏规则**：中文课件各 build 脚本共用这一份（2026-10-20）。

⭐ 为什么有这个文件：中文原稿问题清单第三组 —— 手机 390 下，专题二、四、五的课件页
   被正文表格撑到 460–690px 宽，整页能左右晃，右边被切。英文版拼页时已经加了
   同一条规则（`@media(max-width:1000px){main table,…{display:block;max-width:100%;overflow-x:auto}}`），
   中文页一直没有。
⛔ 不放进 `topic03_page.make_head()`：那是全课共用的 head，改它会让所有用 make_head 的页面
   （包括正在别处施工的专题）下次构建时一起变。这里只给显式 import 它的页面用。
⛔ 一条规则只留一份 —— 各 build 脚本 `from course_narrow_css import CSS` 后插进自己的 <style>，
   不要复制粘贴。专题一课件（topic-01.html）没有生成器，是手工页，那里抄了一份并注明出处。
"""

CSS = """
/* ── 窄屏（course_narrow_css.py，2026-10-20）──────────────────────────
   ① 正文表格在自己框里横向滚动，不再把整页撑宽（跟英文版同一条）。
   ② 节标题那一行是 flex：录音条（.lecmedia，最小 280px）挤在徽章和标题后面会把页面撑出去，
      窄屏下让它另起一行、占满整行。
   ③ 手机上表格已经能在框里横滑了，就别再把格子压扁：公式（td 里的 code）一折就是逐字断行
      （专题四 1.3 形状表每行约 400px 高），数值列「7.32 TiB」折成两行 —— 这两类不折行。 */
@media(max-width:1000px){main table,section table,details table{display:block;max-width:100%;overflow-x:auto}}
@media(max-width:600px){.stn{flex-wrap:wrap}.stn .lecmedia{flex:1 1 100%;min-width:0}
  td code,td.num{white-space:nowrap}}
"""
