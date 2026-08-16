"""真机运行 —— 上 64 卡，把「跑多快」那一半补上。

**真机数据很贵**（占共享集群、几十分钟、抢卡排队），所以这里的第一原则是
**跑一次就要把所有东西留下来**：

  完整日志          → GCS 独立 run 目录（永久，不挂生命周期）
  profile 原始产物  → 同上 + 拉一份到 XProf 的 logdir
  关键值            → Firestore `tpuguru_metal`，**带配置指纹**
  trace 链接        → 报告页最上面那一条

配置指纹是把两半缝起来的针：同一个指纹下，「AOT 说装得下 93.19 GB」和
「真机跑出 670.8 TFLOP/s」会自动并排出现，而不是两笔各存各的。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

log = logging.getLogger("tpuguru.metal")

NS = os.environ.get("TPUGURU_TPU_NS", "priority-dev")
RESERVATION = os.environ.get(
    "TPUGURU_TPU_RESERVATION", "cloudtpu-20260710003900-159478293")
GCS_ROOT = os.environ.get("TPUGURU_METAL_GCS", "gs://chrisya-v7x-us-central1/tpuguru")
XPROF_LOGDIR = Path(os.environ.get("TPUGURU_XPROF_LOGDIR", str(Path.home() / "fp8-prof")))
XPROF_BASE = os.environ.get("TPUGURU_XPROF_URL", "https://cc.higcp.com/xprof/data/plugin/profile/")
PEAK_BF16_PER_CHIP = 2307.0

TOPO_NODES = {"tpu7x-8": (1, "2x2x1"), "tpu7x-32": (8, "2x4x4"), "tpu7x-128": (16, "4x4x4"),
              "tpu7x-256": (32, "4x4x8"), "tpu7x-512": (64, "4x8x8")}


def _run_name(fp: str) -> str:
    return f"tg-{time.strftime('%m%d-%H%M%S')}-{fp[:6]}"


def build_jobset(name: str, params: dict, target: dict, prof: dict, steps: int = 30,
                 profile_from: int = 12, profile_steps: int = 3) -> tuple[str, str]:
    """生成 JobSet YAML。返回 (yaml, gcs_out)。

    ⚠️ `maxRestarts` 写 3 不写 10：SLICE_FAILURE 反复重试会把 DWS 节点整批烧掉。
    """
    topo = target.get("topology", "tpu7x-128")
    nodes, tpu_topo = TOPO_NODES.get(topo, (16, "4x4x4"))
    gcs_out = f"{GCS_ROOT}/{name}"

    args = dict(prof.get("fixed_args", {}))
    pmap = prof.get("param_map", {})
    qmap = prof.get("quant_map", {})
    for k, v in params.items():
        if k in prof.get("tile_expand", {}):
            for real in prof["tile_expand"][k]:
                args[real] = v
            continue
        key = pmap.get(k, k)
        args[key] = qmap.get(str(v), v) if k == "quantization" else (
            "True" if v is True else "False" if v is False else v)
    # 真机跟 AOT 的差别只在这几项：真的跑、真的采 profile、真的写 GCS
    args.pop("compile_topology", None)
    args.pop("compile_topology_num_slices", None)
    args.update({
        "run_name": name, "base_output_directory": gcs_out,
        "steps": steps, "enable_checkpointing": "False",
        "profiler": "xplane", "profiler_steps": profile_steps,
        "skip_first_n_steps_for_profiler": profile_from,
        "upload_all_profiler_results": "True",
    })
    body = " ".join(f'{k}={v}' for k, v in sorted(args.items()) if k != "compile_xla_flags")
    xla = prof.get("default_xla_flags", "")

    cmd = (f'{prof["preamble"]} && export LIBTPU_INIT_ARGS="{xla}" && '
           f'python3 -m {prof["entry"].replace("train_compile", "train")} '
           f'{prof["config"]} {body} 2>&1 | tee /tmp/train.log; '
           f'gcloud storage cp /tmp/train.log {gcs_out}/train.log || true')

    y = f"""apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: {name}
  namespace: {NS}
  labels:
    kueue.x-k8s.io/queue-name: multislice-queue
  annotations:
    alpha.jobset.sigs.k8s.io/exclusive-topology: cloud.google.com/gke-nodepool
spec:
  failurePolicy:
    maxRestarts: 3
  replicatedJobs:
  - name: slice-job
    replicas: 1
    template:
      spec:
        parallelism: {nodes}
        completions: {nodes}
        backoffLimit: 0
        template:
          metadata:
            labels:
              declared-duration-minutes: "60"
          spec:
            restartPolicy: Never
            priorityClassName: very-high
            hostNetwork: true
            dnsPolicy: ClusterFirstWithHostNet
            nodeSelector:
              cloud.google.com/gke-tpu-accelerator: tpu7x
              cloud.google.com/gke-tpu-topology: "{tpu_topo}"
              cloud.google.com/reservation-name: {RESERVATION}
            tolerations:
            - operator: Exists
            containers:
            - name: jax-tpu
              image: {prof["image"]}
              securityContext:
                privileged: true
              resources:
                limits:
                  google.com/tpu: 4
              command: ["bash", "-lc"]
              args:
              - |
                {cmd}
"""
    return y, gcs_out


# ── 日志解析 ────────────────────────────────────────────────────
# 只认这几条，且**每条都写清它是什么口径** —— MFU 的分母、per-chip 的 ×2，
# 都是这类数字最常被搞错的地方。
_STEP = re.compile(r"completed step:\s*(\d+).*?seconds:\s*([\d.]+)", re.I)
_TFLOPS = re.compile(r"TFLOP/s/device:\s*([\d.]+)", re.I)
_LOSS = re.compile(r"loss:\s*([\d.]+),\s*(?:total_weights|lm_loss)", re.I)
_PARAMS = re.compile(r"number parameters:\s*([\d.]+)\s*billion", re.I)


def parse_train_log(text: str, warmup: int = 5) -> dict:
    """从训练日志抽关键值。**跳过前几步**（编译 + warmup），只取稳态。"""
    steps = [(int(m.group(1)), float(m.group(2))) for m in _STEP.finditer(text)]
    tps = [float(m.group(1)) for m in _TFLOPS.finditer(text)]
    steady = [s for s in steps if s[0] >= warmup]
    out: dict = {"steps_seen": len(steps), "steady_steps": len(steady), "warmup_skipped": warmup}
    if steady:
        secs = sorted(s[1] for s in steady)
        out["step_s"] = round(secs[len(secs) // 2], 3)      # 中位数，避开抖动
        out["step_s_min"] = round(secs[0], 3)
        out["step_s_max"] = round(secs[-1], 3)
    if len(tps) > warmup:
        v = sorted(tps[warmup:])
        per_dev = v[len(v) // 2]
        out["tflops_per_device"] = round(per_dev, 1)
        # ★ v7 是 2 device/chip，框架日志一律按 device 报。忘了这个 ×2 就差一倍。
        out["tflops_per_chip"] = round(per_dev * 2, 1)
        out["mfu_pct"] = round(per_dev * 2 / PEAK_BF16_PER_CHIP * 100, 2)
        out["mfu_note"] = f"分母用 BF16 峰值 {PEAK_BF16_PER_CHIP} TFLOP/s/chip"
    m = _LOSS.search(text)
    if m:
        out["first_loss"] = float(m.group(1))
    m = _PARAMS.search(text)
    if m:
        out["params_b"] = float(m.group(1))
    losses = [float(x.group(1)) for x in _LOSS.finditer(text)]
    if len(losses) >= 2:
        out["loss_first"], out["loss_last"] = losses[0], losses[-1]
        if all(abs(l) < 1e-9 for l in losses):
            out["warn"] = ("⚠️ **loss 全是 0** —— 梯度没流动，这一跑的吞吐数字不能信。"
                           "常见于 from-scratch random init + synthetic data 的配置错误。")
    return out


async def _sh(*args, timeout=120) -> tuple[int, str]:
    p = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE,
                                             stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(p.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        p.kill()
        return 124, "timeout"
    return p.returncode, out.decode("utf-8", "replace")


async def collect(name: str, gcs_out: str, fingerprint: str) -> dict:
    """跑完之后把所有东西留下来。真机数据贵，宁可多存。"""
    local = XPROF_LOGDIR / "tpuguru" / name
    local.mkdir(parents=True, exist_ok=True)

    rc, log_text = await _sh("gcloud", "storage", "cat", f"{gcs_out}/train.log", timeout=180)
    if rc != 0:
        rc2, log_text = await _sh("kubectl", "logs", f"job/{name}-slice-job-0",
                                  "-n", NS, "-c", "jax-tpu", "--tail=4000", timeout=180)
    (local / "train.log").write_text(log_text, encoding="utf-8")
    metrics = parse_train_log(log_text)

    # profile 拉到 XProf 的 logdir 下 —— 它按子目录发现 run
    await _sh("gcloud", "storage", "cp", "-r", f"{gcs_out}/{name}/tensorboard",
              str(local), timeout=600)
    xplanes = list(local.rglob("*.xplane.pb"))
    xprof_run, xprof_url = None, None
    if xplanes:
        # run 名是相对 logdir 的路径，形如 tpuguru/<name>/tensorboard/<时间戳>
        rel = xplanes[0].parent.relative_to(XPROF_LOGDIR)
        xprof_run = str(rel)
        xprof_url = f"{XPROF_BASE}?#profile&run={xprof_run}&tag=trace_viewer"

    return {"run_name": name, "fingerprint": fingerprint, "gcs": gcs_out,
            "local_dir": str(local), "metrics": metrics,
            "xprof_run": xprof_run, "xprof_url": xprof_url,
            "xplane_count": len(xplanes),
            "log_bytes": len(log_text.encode()),
            "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
