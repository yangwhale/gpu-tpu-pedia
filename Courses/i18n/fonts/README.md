# Fonts

`ss3-embedded.css` inlines the Latin subset of **Source Sans 3** (Adobe, SIL Open Font License 1.1),
fetched from Google Fonts. `assemble.py` embeds it into every English page so that the reader sees the
same glyphs the layout checks were run with. Source Sans is the design Noto Sans CJK's Latin glyphs derive
from, so English labels keep the widths the figures were designed around.

2026-10-17: the italic face was a static 400 file declared as `font-weight:200 900`, so bold inside
italics (`em b`) rendered at regular weight with no synthesis. Replaced with the variable italic
(wght 200–900, same Latin coverage, 231 code points).
