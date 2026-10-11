#!/usr/bin/env python3
"""c2_ici_bench.py —— 单切片内 ICI 集合通信带宽（C 组）。

放在 runbook/scripts/04-bench/，验收用（04 章「组件验收」C3 卡）。
  ⚠ reduce_scatter 行不要用：实测时打开 C3_HLO=1 看编译后 HLO，psum_scatter(tiled)+tile 被 XLA 改写成 1 个 all-reduce、
  0 个 reduce-scatter，按 RS 公式算出的 busbw 恰好是 AR 的一半（实测）。
  验收只看 all_reduce（以及 c3 的分轴 all_reduce）。

所有进程跑同一个全局程序：一维 mesh 覆盖切片内全部 device（jax.make_mesh，按物理拓扑排序），
shard_map 里做 psum（all-reduce）/ all_gather / psum_scatter（reduce-scatter），多个消息大小。
只有 process 0 打印结果行（前缀 C2_ROW，JSON）。

口径（照 nccl-tests，S = 「整条消息」字节数，bf16）：
  all-reduce     S = 每个 device 的缓冲区大小；algbw = S/t；busbw = algbw × 2(n-1)/n
  all-gather     S = 每个 device 收到的完整输出大小（每 device 贡献 S/n）；busbw = algbw × (n-1)/n
  reduce-scatter S = 每个 device 的完整输入大小（每 device 得到 S/n）；busbw = algbw × (n-1)/n
计时：jit 内 fori_loop 连做 INNER 次（带数据依赖，防 XLA 合并/消除），外层重复 REPS 次取中位数；
  每次迭代带一点胶水运算（缩放 / 切片相加），对大消息可忽略，对小消息会让结果偏低。
"""
import json
import os
import statistics
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

try:
    from jax import shard_map as _sm
    shard_map = _sm
except ImportError:  # 老版本
    from jax.experimental.shard_map import shard_map

INNER = int(os.environ.get("C2_INNER", "5"))
REPS = int(os.environ.get("C2_REPS", "5"))
SIZES_MIB = [int(x) for x in os.environ.get("C2_SIZES_MIB", "4,32,256,1024").split(",")]
P0 = jax.process_index() == 0


def emit(**kw):
    if P0:
        print("C2_ROW " + json.dumps(kw), flush=True)


def smap(f, mesh, in_specs, out_specs):
    try:
        return shard_map(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)
    except TypeError:
        return shard_map(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False)


def timeit(fn, x):
    y = fn(x); jax.block_until_ready(y)
    ts = []
    for _ in range(REPS):
        t = time.perf_counter(); y = fn(x); jax.block_until_ready(y); ts.append(time.perf_counter() - t)
    return statistics.median(ts) / INNER


def main():
    n = jax.device_count()
    mesh = jax.make_mesh((n,), ("x",))
    emit(kind="info", devices=n, processes=jax.process_count(), device_kind=jax.devices()[0].device_kind,
         jax=jax.__version__)
    sh = NamedSharding(mesh, P("x"))
    for mib in SIZES_MIB:
        S = mib << 20
        elems = S // 2  # bf16
        if elems % n:
            continue
        s = elems // n  # 每 device 的分片元素数
        # all-reduce：每 device 持有 S 字节 → 全局数组 [n*elems]
        try:
            def ar_body(x):
                def body(_, v):
                    return jax.lax.psum(v, "x") * jnp.bfloat16(1.0 / n)
                return jax.lax.fori_loop(0, INNER, body, x)
            f = jax.jit(smap(ar_body, mesh, P("x"), P("x")))
            x = jax.jit(lambda: jnp.ones((n * elems,), jnp.bfloat16), out_shardings=sh)()
            t = timeit(f, x); alg = S / t / 1e9
            emit(kind="all_reduce", mib=mib, n=n, us=t * 1e6, algbw_GBps=alg, busbw_GBps=alg * 2 * (n - 1) / n)
            del x
        except Exception as e:
            emit(kind="all_reduce", mib=mib, error=repr(e)[:300])
        # all-gather：每 device 持有 S/n，收齐 S
        try:
            def ag_body(x):
                def body(_, v):
                    y = jax.lax.all_gather(v, "x", tiled=True)
                    return y[:s] + y[-s:]
                return jax.lax.fori_loop(0, INNER, body, x)
            f = jax.jit(smap(ag_body, mesh, P("x"), P("x")))
            x = jax.jit(lambda: jnp.ones((elems,), jnp.bfloat16), out_shardings=sh)()
            t = timeit(f, x); alg = S / t / 1e9
            emit(kind="all_gather", mib=mib, n=n, us=t * 1e6, algbw_GBps=alg, busbw_GBps=alg * (n - 1) / n)
            del x
        except Exception as e:
            emit(kind="all_gather", mib=mib, error=repr(e)[:300])
        # reduce-scatter：每 device 持有 S，得到 S/n
        try:
            def rs_body(x):
                def body(_, v):
                    y = jax.lax.psum_scatter(v, "x", tiled=True)
                    return jnp.tile(y, n) * jnp.bfloat16(1.0 / n)
                return jax.lax.fori_loop(0, INNER, body, x)
            f = jax.jit(smap(rs_body, mesh, P("x"), P("x")))
            x = jax.jit(lambda: jnp.ones((n * elems,), jnp.bfloat16), out_shardings=sh)()
            t = timeit(f, x); alg = S / t / 1e9
            emit(kind="reduce_scatter", mib=mib, n=n, us=t * 1e6, algbw_GBps=alg, busbw_GBps=alg * (n - 1) / n)
            del x
        except Exception as e:
            emit(kind="reduce_scatter", mib=mib, error=repr(e)[:300])


if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    except Exception as e:
        emit(kind="error", error=repr(e)[:500])
    emit(kind="done", seconds=round(time.time() - t0, 1))
