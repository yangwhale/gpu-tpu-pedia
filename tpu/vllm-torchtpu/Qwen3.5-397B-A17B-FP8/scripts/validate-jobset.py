#!/usr/bin/env python3
"""JobSet 三层校验。缺任何一层都可能让你在提交后才发现问题。

用法: python3 validate-jobset.py jobset.yaml [--context CTX]

第 3 层是重点：JobSet 的 `kubectl apply --dry-run=server` **只校验 JobSet CRD 本身，
不校验它将来生成的 Job**。真出问题时 dry-run 一路绿灯，真因只藏在
`kubectl logs -n jobset-system -l control-plane=controller-manager` 里。
实测这一个疏漏废掉过一整夜 9 个任务。
"""
import argparse
import subprocess
import sys
import tempfile

import yaml


class NoDupLoader(yaml.SafeLoader):
    pass


def _no_dup(loader, node, deep=False):
    m = {}
    for k, v in node.value:
        key = loader.construct_object(k, deep=deep)
        if key in m:
            raise ValueError(f"重复键: {key}")
        m[key] = loader.construct_object(v, deep=deep)
    return m


NoDupLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_dup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--context", default=None)
    a = ap.parse_args()
    kube = ["kubectl"] + (["--context", a.context] if a.context else [])

    # ── 第 1 层：严格 YAML 解析，抓重复键 ──
    # 两个 limits 块时后者会覆盖前者，google.com/tpu 静默丢限额。
    # 这类错误 --dry-run 完全看不出来，因为发到 API server 的已经是「合并后」的结果。
    try:
        d = yaml.load(open(a.path, encoding="utf-8"), Loader=NoDupLoader)
    except ValueError as e:
        print(f"✗ 第 1 层: {e}")
        sys.exit(1)
    c = d["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]["containers"][0]
    r = c["resources"]
    for side in ("limits", "requests"):
        if r.get(side, {}).get("google.com/tpu") is None:
            print(f"✗ 第 1 层: {side} 里没有 google.com/tpu")
            sys.exit(1)
    print(f"✓ 第 1 层 无重复键，TPU limits={r['limits']['google.com/tpu']} "
          f"requests={r['requests']['google.com/tpu']}")

    # ── 第 2 层：JobSet CRD 服务端校验 ──
    # 必须先改名。同名对象已存在时，dry-run apply 会走 patch 路径，
    # 撞上 JobSet 的 immutable 字段报 Forbidden —— 那是「对象已存在」不是「配置有错」，
    # 不改名的话第二次校验就永远过不去。
    d2 = yaml.safe_load(open(a.path, encoding="utf-8"))
    d2["metadata"]["name"] = d2["metadata"]["name"] + "-validate-only"
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                     encoding="utf-8") as f:
        yaml.safe_dump(d2, f)
        tmp2 = f.name
    p = subprocess.run(kube + ["apply", "-f", tmp2, "--dry-run=server"],
                       capture_output=True, text=True)
    if p.returncode:
        print(f"✗ 第 2 层: {p.stderr.strip()[:300]}")
        sys.exit(1)
    print("✓ 第 2 层 JobSet CRD 通过")

    # ── 第 3 层：把 pod template 抽成独立 Job 再校验 ──
    # 只有这层真正校验 pod spec（资源量纲、volume 引用、字段合法性）。
    t = d["spec"]["replicatedJobs"][0]["template"]
    job = {"apiVersion": "batch/v1", "kind": "Job",
           "metadata": {"name": "validate-only"}, "spec": dict(t["spec"])}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                     encoding="utf-8") as f:
        yaml.safe_dump(job, f)
        tmp = f.name
    p = subprocess.run(kube + ["apply", "-f", tmp, "--dry-run=server"],
                       capture_output=True, text=True)
    if p.returncode:
        print(f"✗ 第 3 层（pod spec）: {p.stderr.strip()[:300]}")
        sys.exit(1)
    print("✓ 第 3 层 pod spec 通过")
    print("三层全过，可以提交")


if __name__ == "__main__":
    main()
