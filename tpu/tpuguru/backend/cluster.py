"""训练集群状态 —— 页面顶栏那盏灯。

判据取自 tpu-v7-cluster 的经验：**看队列不看节点**。
「PROVISIONING、0 节点」在这个集群里通常意味着 Kueue 还没 admit 你，
而不是抢不到容量；反过来「Kueue 记账满了」也不等于机器忙。
所以三个数都要给：保底、cohort 空闲、我们已用。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time

log = logging.getLogger("tpuguru.cluster")

PROJECT = os.environ.get("TPUGURU_TPU_PROJECT", "cloud-tpu-shared-capacity")
CLUSTER = os.environ.get("TPUGURU_TPU_CLUSTER", "bodaborg-tpu7x-nap")
REGION = os.environ.get("TPUGURU_TPU_REGION", "us-central1")
NS = os.environ.get("TPUGURU_TPU_NS", "priority-dev")
TTL = 60          # 秒。别每次刷页面都去打 kubectl

_cache: dict = {"at": 0.0, "data": None}


async def _kubectl(*args, timeout=25) -> str:
    p = await asyncio.create_subprocess_exec(
        "kubectl", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    try:
        out, _ = await asyncio.wait_for(p.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        p.kill()
        raise
    return out.decode("utf-8", "replace")


async def _probe() -> dict:
    cq = json.loads(await _kubectl("get", "clusterqueue", "-o", "json"))
    quotas, used, cohort_used, cohort_nominal = {}, {}, 0, 0
    for it in cq.get("items", []):
        name = it["metadata"]["name"]
        nom = 0
        for g in (it.get("spec", {}).get("resourceGroups") or []):
            for f in (g.get("flavors") or []):
                for r in (f.get("resources") or []):
                    if "tpu" in r.get("name", ""):
                        nom += int(str(r.get("nominalQuota", 0)).rstrip("m") or 0)
        u = 0
        for f in (it.get("status", {}).get("flavorsUsage") or []):
            for r in (f.get("resources") or []):
                if "tpu" in r.get("name", ""):
                    u += int(str(r.get("total", 0)).rstrip("m") or 0)
        quotas[name], used[name] = nom, u
        cohort_nominal += nom
        cohort_used += u

    ours_quota = quotas.get(NS, 0)
    ours_used = used.get(NS, 0)
    free = max(cohort_nominal - cohort_used, 0)

    pods = await _kubectl("get", "pods", "-n", NS, "--no-headers")
    running = sum(1 for l in pods.splitlines() if " Running " in f" {l} ")

    return {"quota": ours_quota, "used": ours_used, "cohort_free": free,
            "cohort_total": cohort_nominal, "running_pods": running,
            "namespace": NS, "cluster": CLUSTER}


def _verdict(d: dict, want: int) -> dict:
    """绿 / 黄 / 红 —— 判据写在 why 里，别让人对着一盏灯猜。"""
    free, quota, used = d["cohort_free"], d["quota"], d["used"]
    # 灯上只放一句话 —— cohort / 保底 / 借用这些账，塞进 tooltip 就够了
    if free >= want:
        return {"light": "green", "text": f"{want} 卡可用",
                "why": f"cohort 当前空闲 {free} 芯片，够你要的 {want}。几十秒就能起来。"}
    if used + want <= quota:
        return {"light": "amber", "text": f"{want} 卡需等",
                "why": f"空闲只有 {free}，但「已用 {used} + 申请 {want}」没超我们 {quota} 的保底 —— "
                       f"Kueue 会把别人借走的抢回来，通常几分钟。"}
    short = used + want - quota
    if short <= free:
        return {"light": "amber", "text": f"{want} 卡需等",
                "why": f"超出的 {short} ≤ cohort 空闲 {free}，能借但要等，可能十几分钟。"}
    return {"light": "red", "text": f"{want} 卡要不到",
            "why": f"超出保底 {short} 芯片，而 cohort 只空闲 {free}。"
                   f"改小规模、等别人释放，或挑 peak 窗口（港时 04:00–16:00 工作日）。"}


async def status(want: int = 64) -> dict:
    now = time.time()
    if _cache["data"] and now - _cache["at"] < TTL:
        d = dict(_cache["data"])
    else:
        try:
            d = await _probe()
            _cache.update(at=now, data=d)
        except Exception as e:  # noqa: BLE001
            log.warning("集群探测失败: %s", e)
            return {"ok": False, "light": "grey", "text": "连不上集群",
                    "why": f"{e}。可能是 kubeconfig 过期 —— "
                           f"gcloud container clusters get-credentials {CLUSTER} "
                           f"--region={REGION} --project={PROJECT}"}
    d["ok"] = True
    d["want"] = want
    d.update(_verdict(d, want))
    return d
