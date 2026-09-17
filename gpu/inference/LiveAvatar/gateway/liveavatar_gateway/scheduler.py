"""槽位调度 —— 决定这一路派给哪张卡。

模型是 pull：worker 主动注册＋长轮询取活。这样 GPU 机器
**不需要对外开端口**，也能在 NAT/不同 VPC 后面，Spot 被回收后重新注册即可。
控制面只记账，不主动连 worker。
"""
from __future__ import annotations

from dataclasses import dataclass

from .store import Store, Worker


class NoCapacity(Exception):
    """所有卡都忙。调用方看到 429 + Retry-After。"""


@dataclass
class Capacity:
    total: int
    used: int

    @property
    def free(self) -> int:
        return max(0, self.total - self.used)


class Scheduler:
    def __init__(self, store: Store, *, heartbeat_timeout_s: float):
        self._store = store
        self._hb_timeout = heartbeat_timeout_s

    def capacity(self) -> Capacity:
        workers = self._store.live_workers(self._hb_timeout)
        counts = self._store.active_counts()
        total = sum(w.capacity for w in workers)
        used = sum(counts.get(w.worker_id, 0) for w in workers)
        return Capacity(total=total, used=used)

    def pick_worker(self) -> Worker:
        """最空闲优先。相同空闲度下取 worker_id 最小的，保证可复现。"""
        workers = self._store.live_workers(self._hb_timeout)
        if not workers:
            raise NoCapacity("没有存活的 worker")
        counts = self._store.active_counts()
        free = [(w.capacity - counts.get(w.worker_id, 0), w) for w in workers]
        free = [(n, w) for n, w in free if n > 0]
        if not free:
            raise NoCapacity("所有槽位都忙")
        free.sort(key=lambda t: (-t[0], t[1].worker_id))
        return free[0][1]

    def sweep_dead_workers(self) -> list[str]:
        """心跳丢失的 worker：关掉它身上的会话并摘掉它。

        ⚠️ 不删会话记录，只置 closed —— 留痕才查得出「这一路是怎么没的」。
        """
        live = {w.worker_id for w in self._store.live_workers(self._hb_timeout)}
        dropped: list[str] = []
        counts = self._store.active_counts()
        for worker_id in counts:
            if worker_id not in live:
                self._store.close_sessions_of_worker(worker_id)
                self._store.drop_worker(worker_id)
                dropped.append(worker_id)
        return dropped
