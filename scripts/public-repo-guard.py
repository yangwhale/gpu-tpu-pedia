#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""公开仓库闸门 —— 提交前拦下不该出去的字样。

**这个仓库是公开的 GitHub 仓库。** 一旦 push，内容就进了 git 历史，
删掉也还在，所以唯一有效的防线是**在 commit 之前拦住**。

为什么现在才有：2026-08-31 在 `tpu/Hunyuan3-295B-Pretraining/AOT-COMPILE.md`
里发现一条内部构建路径，**已经躺了很久**。当时仓库里唯一的闸门是
`Courses/tools/tpu-micro/gate.py`，而它只在生成**一张** HTML 时跑 ——
仓库里其余几百个文件谁都没管。一道只盖住 1% 的闸门，
比没有闸门更危险：它让人以为这件事已经有人管了。

⭐ **禁字表不在这里，在 `Courses/tools/tpu-micro/gate.py` 的 `_FORBIDDEN`。**
两份表迟早会漂 —— 一处补了另一处没补，然后你以为两道防线，其实是一道。
所以这里直接 import 它，**单一来源**。

用法：

    python3 scripts/public-repo-guard.py            # 扫已 staged 的文件（pre-commit 用）
    python3 scripts/public-repo-guard.py --all      # 扫全仓（第一次接手时跑一遍）
    python3 scripts/public-repo-guard.py a.md b.py  # 扫指定文件

装成 pre-commit 钩子（钩子不进版本库，**每台机器要各装一次**）：

    python3 scripts/public-repo-guard.py --install-hook

退出码：0 干净 / 1 有命中 / 2 用不了（例如找不到禁字表）。
"""
import io
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Courses", "tools", "tpu-micro"))

try:
    from gate import _FORBIDDEN, PAGE_ONLY           # noqa: E402  单一来源
except Exception as e:                               # pragma: no cover
    sys.stderr.write("❌ 读不到禁字表 Courses/tools/tpu-micro/gate.py：%s\n" % e)
    sys.exit(2)

# 只扫文本。二进制（图片、字体）不看 —— 它们不是泄漏的载体，
# 而且 400 MB 的产物扫一遍会让钩子慢到有人想绕过它。
TEXT_EXT = {".md", ".html", ".htm", ".py", ".sh", ".txt", ".json", ".yaml",
            ".yml", ".csv", ".js", ".css", ".ts", ".toml", ".cfg", ".ini"}

# ⛔ 这两个文件**就是禁字表本身**，不能扫，否则它永远匹配自己。
# （同一个坑在别处踩过：在自己的日志里 grep 错误串，一定会自匹配。）
SELF = {
    os.path.join("Courses", "tools", "tpu-micro", "gate.py"),
    os.path.join("scripts", "public-repo-guard.py"),
}

SKIP_DIRS = {".git", "node_modules", "__pycache__", "out", "public"}


def _staged():
    out = subprocess.run(["git", "diff", "--cached", "--name-only",
                          "--diff-filter=ACMR"],
                         cwd=ROOT, capture_output=True, text=True)
    return [p for p in out.stdout.splitlines() if p.strip()]


def _walk_all():
    hit = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            hit.append(os.path.relpath(full, ROOT))
    return hit


def scan(rel_paths):
    """返回 [(相对路径, 行号, 命中字样, 说明, 整行), ...]。"""
    found = []
    for rel in rel_paths:
        if rel in SELF:
            continue
        if os.path.splitext(rel)[1].lower() not in TEXT_EXT:
            continue
        full = os.path.join(ROOT, rel)
        if not os.path.isfile(full):
            continue
        try:
            text = io.open(full, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for pat, desc in _FORBIDDEN:
            if desc in PAGE_ONLY:      # 只对生成出来的公开页面成立，见 gate.PAGE_ONLY
                continue
            for m in re.finditer(pat, text):
                ln = text.count("\n", 0, m.start()) + 1
                line = text.splitlines()[ln - 1] if ln else ""
                found.append((rel, ln, m.group(0), desc, line.strip()[:160]))
    return found


def main(argv):
    if "--install-hook" in argv:
        hook = os.path.join(ROOT, ".git", "hooks", "pre-commit")
        io.open(hook, "w", encoding="utf-8").write(
            "#!/bin/sh\n"
            "# 由 scripts/public-repo-guard.py --install-hook 生成。\n"
            "exec python3 \"$(git rev-parse --show-toplevel)\""
            "/scripts/public-repo-guard.py\n")
        os.chmod(hook, 0o755)
        print("✅ 已装 %s" % hook)
        print("   （钩子不进版本库 —— 换一台机器要再跑一次这条命令）")
        return 0

    if "--all" in argv:
        paths = _walk_all()
    else:
        paths = [a for a in argv[1:] if not a.startswith("-")] or _staged()

    hits = scan(paths)
    if not hits:
        print("✅ 公开仓库闸门：%d 个文件，没有命中" % len(paths))
        return 0

    sys.stderr.write("\n⛔ 公开仓库闸门拦下 %d 处 —— 这个仓库 push 出去就删不掉了\n\n"
                     % len(hits))
    for rel, ln, tok, desc, line in hits:
        sys.stderr.write("  %s:%d  【%s】%s\n      %s\n" % (rel, ln, desc, tok, line))
    sys.stderr.write(
        "\n改掉，或者确实是误报就去 Courses/tools/tpu-micro/gate.py 调规则。\n"
        "急着提交可以 git commit --no-verify 绕过 —— "
        "但绕过一次就等于这道闸门不存在，想清楚再绕。\n\n")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
