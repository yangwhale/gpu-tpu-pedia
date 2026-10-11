#!/usr/bin/env python3
"""05-avoid-manual-slices.py —— 给 Kueue 路径的 JobSet 注入 nodeAffinity，避开被「手工 Slice」占着的节点。

为什么：手工建的 Slice（不经 Kueue，没有 accelerator.gke.io/owner-workload-name 注解）对 Kueue/TAS 不可见。
  TAS 仍把那些 cube 当空闲选进来 → GKE Slice Controller 报
  `step AttachMIGsToMMIG failed: ... cannot be attached to the parent workload group because it has an active
   sub slice in it with external slice ID <手工片名>-<hash8> ...` → SliceCreationFailed 反复重试、永不 ACTIVE。
  实测：4x8x8 / 8x8x8 / 8x8x16 三次都因此 10 min 超时；
  加上本脚本的 NotIn 亲和后 8x16x16（2,048 芯）与 10+ × 4x8x8（数千芯）一次过。

做法：对每个 JobSet 的每个 replicatedJob 的 pod 模板，追加
  requiredDuringSchedulingIgnoredDuringExecution: cloud.google.com/gke-tpu-slice NotIn [手工片名…]
  （NotIn 对「没有这个标签」的节点也成立，所以只排除挂着这些手工片标签的节点；TAS 选 cube 时会遵守 required 亲和。）
  已有的 nodeSelectorTerms 会逐条追加这一条 matchExpression（terms 之间是 OR，所以每条都要加），而不是覆盖。

用法：
  ./04-gen-jobset.sh 4x8x8 --slices N --multi-nic > js.yaml
  ./05-avoid-manual-slices.py js.yaml js.yaml                 # 不给名字：自动取集群里所有没有 owner-workload-name 注解的 Slice
  ./05-avoid-manual-slices.py js.yaml js.yaml <手工片名1> <手工片名2>  # 或显式给出手工片名
  输入/输出可以用 - 表示 stdin/stdout。需要 python3 + PyYAML；自动发现要 kubectl 读 slices 的权限。

局限：
  - 只能避开「提交那一刻」已存在的手工 Slice；提交后才建的手工片仍会撞上。根本解法：同一集群别混用手工 Slice 和 Kueue，巡检用的小切片用完即删（经验教训第 33 条）。
  - 手工片的节点标签 gke-tpu-slice 只在 ACTIVE 后才打上；还在 ACTIVATING 的手工片避不开。
"""
import json
import subprocess
import sys

import yaml

KEY = "cloud.google.com/gke-tpu-slice"
OWNER = "accelerator.gke.io/owner-workload-name"


def manual_slices():
    out = subprocess.run(["kubectl", "get", "slices.accelerator.gke.io", "-o", "json"],
                         check=True, capture_output=True, text=True).stdout
    return sorted(i["metadata"]["name"] for i in json.loads(out)["items"]
                  if OWNER not in (i["metadata"].get("annotations") or {}))


def inject(pod_spec, names):
    expr = {"key": KEY, "operator": "NotIn", "values": names}
    req = (pod_spec.setdefault("affinity", {}).setdefault("nodeAffinity", {})
           .setdefault("requiredDuringSchedulingIgnoredDuringExecution", {}))
    terms = req.setdefault("nodeSelectorTerms", [])
    if not terms:
        terms.append({"matchExpressions": [expr]})
        return
    for t in terms:
        t.setdefault("matchExpressions", []).append(dict(expr))


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src, dst, names = sys.argv[1], sys.argv[2], sys.argv[3:]
    if not names:
        names = manual_slices()
        print(f"# 自动发现手工 Slice {len(names)} 个：{' '.join(names) or '（无）'}", file=sys.stderr)
    docs = list(yaml.safe_load_all(sys.stdin if src == "-" else open(src)))
    n = 0
    if names:
        for d in docs:
            if not d or d.get("kind") != "JobSet":
                continue
            for rj in d["spec"]["replicatedJobs"]:
                inject(rj["template"]["spec"]["template"]["spec"], names)
                n += 1
    print(f"# 已给 {n} 个 replicatedJob 注入 {KEY} NotIn（{len(names)} 个名字）", file=sys.stderr)
    out = sys.stdout if dst == "-" else open(dst, "w")
    yaml.safe_dump_all(docs, out, sort_keys=False, width=10000, allow_unicode=True)


if __name__ == "__main__":
    main()
