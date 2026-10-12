#!/usr/bin/env python3
"""c1_chip_bench.py —— 单芯片算力 / HBM / host<->device 微基准（C 组）。

放在 runbook/scripts/04-bench/，验收用。info 行带上
  libtpu_init_args / libtpu 版本 / p_state —— P7 档实测证明同一颗芯片 bf16 单 device 默认档 868、P7 995 TFLOP/s，
  算力数字不带功耗档就没法判读。判读阈值见 thresholds.csv。

每个进程（= 一台 host，4 chip = 8 JAX device）只用自己的本地 device，不做任何跨 host 通信，
所以在多 host 切片里每个 pod 各测各的；每行输出带 host 名，前缀 C1_ROW（JSON）。

测什么：
  1. matmul TFLOP/s：bf16 / fp8(e4m3fn, 输出 bf16) × N ∈ {4096, 8192, 16384, 32768}
     - single：只在 local_devices()[0] 上跑（= 1 个 device = 半个 chip）
     - all   ：8 个本地 device 同时各算一份（batched einsum，按 device 维分片），报每 chip = 总 FLOP / 时间 / chip 数
     jit 里 fori_loop 连乘 INNER 次（x = x @ b），消掉派发开销；FLOP = 2 N^3 × INNER。
     峰值口径：bf16 每 chip 2,307 / 每 device 1,153.5；fp8 每 chip 4,614 / 每 device 2,307（官方规格）。
  2. HBM 带宽：y = x * c + 1 （读一遍写一遍），字节 = 2 × 数组大小；single 与 all 两种。
     参考：每 chip 7.38 TB/s（官方规格），每 device ≈ 3.69 TB/s [推断 = 7.38 / 2]。
  3. host<->device：1 GiB float32 numpy → device_put 到 1 个 device（H2D）；新算出的 device 数组 → np.asarray（D2H）；
     另测 8 个 device 各 1 GiB 同时 device_put 的聚合 H2D。
"""
import json
import os
import socket
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

HOST = socket.gethostname()
PEAK_DEV = {"bf16": 1153.5, "fp8": 2307.0}
chips = 4
INNER = int(os.environ.get("C1_INNER", "10"))
REPS = int(os.environ.get("C1_REPS", "5"))


def emit(**kw):
    kw["host"] = HOST
    print("C1_ROW " + json.dumps(kw), flush=True)


def timeit(fn, *args):
    out = fn(*args); jax.block_until_ready(out)  # 编译 + 预热
    ts = []
    for _ in range(REPS):
        t = time.perf_counter(); out = fn(*args); jax.block_until_ready(out); ts.append(time.perf_counter() - t)
    return statistics.median(ts), min(ts)


def matmul_bench():
    global chips
    devs = jax.local_devices(); nd = len(devs)
    chips = len({tuple(d.coords) for d in devs}) if hasattr(devs[0], "coords") else nd // 2
    lia = os.environ.get("LIBTPU_INIT_ARGS", "")
    m = [a.split("=", 1)[1] for a in lia.split() if a.startswith("--xla_tpu_dvfs_p_state=")]
    try:
        from importlib.metadata import version as _v
        libtpu_ver = _v("libtpu")
    except Exception:
        libtpu_ver = "?"
    emit(kind="info", libtpu_init_args=lia, libtpu=libtpu_ver, p_state=(m[-1] if m else "default"),
         local_devices=nd, chips=chips, device_kind=devs[0].device_kind,
         cores=[getattr(d, "core_on_chip", None) for d in devs], coords=[list(getattr(d, "coords", [])) for d in devs],
         jax=jax.__version__)
    mesh = Mesh(np.array(devs), ("d",))
    dtypes = {"bf16": jnp.bfloat16}
    try:
        dtypes["fp8"] = jnp.float8_e4m3fn
    except AttributeError:
        pass
    for name, dt in dtypes.items():
        for n in (4096, 8192, 16384, 32768):
            flop = 2 * n ** 3 * INNER

            def chain(a, b):
                def body(_, x):
                    return jnp.dot(x, b, preferred_element_type=jnp.bfloat16).astype(dt)
                return jax.lax.fori_loop(0, INNER, body, a)

            try:
                # single device
                k1, k2 = jax.random.split(jax.random.key(0))
                a = jax.device_put((jax.random.normal(k1, (n, n), jnp.float32) * 0.01).astype(dt), devs[0])
                b = jax.device_put((jax.random.normal(k2, (n, n), jnp.float32) * 0.01).astype(dt), devs[0])
                med, best = timeit(jax.jit(chain), a, b)
                tf = flop / med / 1e12
                emit(kind="matmul", scope="single_device", dtype=name, n=n, ms=med * 1e3, tflops_dev=tf,
                     tflops_dev_best=flop / best / 1e12, pct_peak=100 * tf / PEAK_DEV[name])
                del a, b
                # all local devices concurrently (batched, sharded on device dim)
                sh = NamedSharding(mesh, P("d"))

                mk = jax.jit(lambda key: (jax.random.normal(key, (nd, n, n), jnp.float32) * 0.01).astype(dt),
                             out_shardings=sh)
                A = mk(k1); B = mk(k2)

                def bchain(A, B):
                    def body(_, X):
                        return jnp.einsum("bij,bjk->bik", X, B, preferred_element_type=jnp.bfloat16).astype(dt)
                    return jax.lax.fori_loop(0, INNER, body, A)
                med, best = timeit(jax.jit(bchain, in_shardings=(sh, sh), out_shardings=sh), A, B)
                tot = flop * nd / med / 1e12
                emit(kind="matmul", scope="all_local", dtype=name, n=n, ms=med * 1e3, tflops_host=tot,
                     tflops_chip=tot / chips, tflops_dev=tot / nd, pct_peak=100 * (tot / nd) / PEAK_DEV[name])
                del A, B
            except Exception as e:  # graceful：某个 dtype/尺寸不支持时记下来继续
                emit(kind="matmul", dtype=name, n=n, error=repr(e)[:300])


def hbm_bench():
    devs = jax.local_devices(); nd = len(devs)
    mesh = Mesh(np.array(devs), ("d",)); sh = NamedSharding(mesh, P("d"))
    for gib in (1, 4):
        n = gib * (1 << 30) // 4  # float32 元素数
        f = jax.jit(lambda x: x * 1.0001 + 1.0)
        x = jax.device_put(jnp.zeros((n,), jnp.float32), devs[0])
        med, best = timeit(f, x)
        emit(kind="hbm", scope="single_device", gib=gib, ms=med * 1e3, tbps_dev=2 * n * 4 / med / 1e12,
             tbps_dev_best=2 * n * 4 / best / 1e12)
        del x
        # 注意：不能用 (nd, n) —— 每个分片是 (1, n)，TPU 的 (8,128) tile 会把第二维 1 补成 8，实测带宽只剩 ~40%
        X = jax.jit(lambda: jnp.zeros((nd, n // 1024, 1024), jnp.float32), out_shardings=sh)()
        fa = jax.jit(lambda x: x * 1.0001 + 1.0, in_shardings=sh, out_shardings=sh)
        med, best = timeit(fa, X)
        tot = 2 * n * 4 * nd / med / 1e12
        emit(kind="hbm", scope="all_local", gib=gib, ms=med * 1e3, tbps_host=tot, tbps_chip=tot / chips, tbps_dev=tot / nd)
        del X


def h2d_bench():
    devs = jax.local_devices()
    nbytes = 1 << 30
    host = np.random.default_rng(0).standard_normal(nbytes // 4, dtype=np.float32)
    # H2D single
    ts = []
    for _ in range(REPS + 1):
        t = time.perf_counter(); y = jax.device_put(host, devs[0]); y.block_until_ready(); ts.append(time.perf_counter() - t); del y
    med = statistics.median(ts[1:])
    emit(kind="h2d", scope="single_device", gib=1, ms=med * 1e3, gbps=nbytes / med / 1e9)
    # D2H single（每次新算一个数组，避免 np.asarray 命中缓存）
    base = jax.device_put(host, devs[0]); ts = []
    for i in range(REPS + 1):
        z = base + float(i); z.block_until_ready()
        t = time.perf_counter(); _ = np.asarray(z); ts.append(time.perf_counter() - t); del z
    med = statistics.median(ts[1:])
    emit(kind="d2h", scope="single_device", gib=1, ms=med * 1e3, gbps=nbytes / med / 1e9)
    del base
    # H2D 聚合：8 个 device 各 1 GiB 同时发
    ts = []
    for _ in range(REPS + 1):
        t = time.perf_counter(); ys = [jax.device_put(host, d) for d in devs]; jax.block_until_ready(ys)
        ts.append(time.perf_counter() - t); del ys
    med = statistics.median(ts[1:])
    emit(kind="h2d", scope="all_local_loop", gib_per_dev=1, ms=med * 1e3, gbps_host=len(devs) * nbytes / med / 1e9)
    del host
    # H2D / D2H 聚合（一次调用，按 device 维分片）：[nd, rows, 1024] float32，每 device 1 GiB
    nd = len(devs); mesh = Mesh(np.array(devs), ("d",)); sh = NamedSharding(mesh, P("d"))
    big = np.ones((nd, nbytes // 4 // 1024, 1024), np.float32)
    ts = []
    for _ in range(REPS + 1):
        t = time.perf_counter(); Y = jax.device_put(big, sh); Y.block_until_ready(); ts.append(time.perf_counter() - t); del Y
    med = statistics.median(ts[1:])
    emit(kind="h2d", scope="all_local_sharded", gib_per_dev=1, ms=med * 1e3, gbps_host=nd * nbytes / med / 1e9)
    Y = jax.device_put(big, sh); ts = []
    for i in range(REPS + 1):
        Z = Y + float(i); Z.block_until_ready()
        t = time.perf_counter(); _ = np.asarray(Z); ts.append(time.perf_counter() - t); del Z
    med = statistics.median(ts[1:])
    emit(kind="d2h", scope="all_local_sharded", gib_per_dev=1, ms=med * 1e3, gbps_host=nd * nbytes / med / 1e9)


if __name__ == "__main__":
    t0 = time.time()
    for fn in (matmul_bench, hbm_bench, h2d_bench):
        try:
            fn()
        except Exception as e:
            emit(kind="error", stage=fn.__name__, error=repr(e)[:500])
    emit(kind="done", seconds=round(time.time() - t0, 1))
