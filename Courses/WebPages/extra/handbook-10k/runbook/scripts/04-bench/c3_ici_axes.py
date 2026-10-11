#!/usr/bin/env python3
"""c3_ici_axes.py —— 片内 ICI：按物理轴建多维 mesh，分轴测集合通信（C 组跟进）。

放在 runbook/scripts/04-bench/，验收用（04 章「组件验收」C3 卡）。
  ⚠ reduce_scatter 行不要用：实测时打开 C3_HLO=1 看编译后 HLO，psum_scatter(tiled)+tile 被 XLA 改写成 1 个 all-reduce、
  0 个 reduce-scatter，按 RS 公式算出的 busbw 恰好是 AR 的一半（实测）。
  验收只看 all_reduce（以及 c3 的分轴 all_reduce）。

c2_ici_bench.py 用一维 mesh（jax.make_mesh((n,))）对全部 device 做 psum，会看到 4x4x8 / 4x8x8 的 1 GiB all-reduce
busbw 只有 4x4x4 / 8x8x8 的 40–50%。本脚本回答两件事：
  1) 每条物理轴（x / y / z，以及同一芯片上两个 TensorCore 的 c 轴）各自的集合通信带宽是多少——看是不是某条轴本身慢；
  2) 全 device all-reduce 的几种写法：
       all_1d   ：一维 mesh psum（= c2_ici_bench.py 的写法，对照）
       all_nd   ：按物理坐标排好的 4 维 mesh，psum 同时跨全部轴（replica group 与 all_1d 相同，只是 device 顺序不同）
       hier     ：手工分层：依次 psum_scatter 过 c→z→y→x（每步把数据切小），再反向 all_gather——3D torus 上的经典分解
只有 process 0 打印结果（前缀 C3_ROW，JSON）。

口径（同 c2，照 nccl-tests，S = 每 device 的缓冲区字节数，bf16；k = 参与这次集合通信的组大小）：
  all-reduce     busbw = S/t × 2(k−1)/k
  all-gather     S = 每 device 收到的完整输出；busbw = S/t × (k−1)/k
  reduce-scatter S = 每 device 的完整输入；  busbw = S/t × (k−1)/k
分轴测时，所有 device 同时在各自的组里做（例如 psum over z：4x4x8 上 16×2 个组各 8 个 device 同时跑），
即训练里「沿某一维做集合通信」的真实情形。
"""
import json
import os
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map

INNER = int(os.environ.get("C3_INNER", "5"))
REPS = int(os.environ.get("C3_REPS", "5"))
SIZES_MIB = [int(x) for x in os.environ.get("C3_SIZES_MIB", "64,256,1024").split(",")]
P0 = jax.process_index() == 0


def emit(**kw):
    if P0:
        kw["ts"] = round(time.time(), 1)
        print("C3_ROW " + json.dumps(kw), flush=True)


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


def physical_mesh():
    """按 device.coords (+ core_on_chip) 排成 (X, Y, Z, C) 数组。"""
    devs = jax.devices()
    coords = [tuple(d.coords) for d in devs]
    cores = [getattr(d, "core_on_chip", 0) for d in devs]
    mins = [min(c[i] for c in coords) for i in range(3)]
    dims = [max(c[i] for c in coords) - mins[i] + 1 for i in range(3)]
    ncore = max(cores) + 1
    arr = np.empty(dims + [ncore], dtype=object)
    for d, c, k in zip(devs, coords, cores):
        arr[c[0] - mins[0], c[1] - mins[1], c[2] - mins[2], k] = d
    assert all(a is not None for a in arr.flat), "coords 不是满的长方体"
    return arr, dims, ncore, devs


def run_op(mesh, op, axes, S, k, hier_order=None):
    """在 mesh 上沿 axes（tuple）做 op，所有 device 的缓冲区 S 字节。返回秒/次。"""
    n = mesh.devices.size
    names = mesh.axis_names
    spec = P(names)  # 数据按全部轴切：每 device 一块
    sh = NamedSharding(mesh, spec)
    elems = S // 2
    if op in ("all_reduce", "hier"):
        per = elems
    elif op == "all_gather":
        per = elems // k
    else:  # reduce_scatter
        per = elems
    if per % 1024 or elems % k:
        return None
    ax = (axes if len(axes) > 1 else axes[0]) if axes else None

    if op == "all_reduce":
        def body(_, v):
            return jax.lax.psum(v, ax) * jnp.bfloat16(1.0 / k)
    elif op == "all_gather":
        s = per
        def body(_, v):
            y = jax.lax.all_gather(v, ax, tiled=True)
            return y[:s] + y[-s:]
    elif op == "reduce_scatter":
        def body(_, v):
            y = jax.lax.psum_scatter(v, ax, tiled=True)
            return jnp.tile(y, k) * jnp.bfloat16(1.0 / k)
    else:  # hier：reduce-scatter 逐轴，再反向 all-gather
        order = hier_order
        def body(_, v):
            for a in order:
                v = jax.lax.psum_scatter(v, a, tiled=True)
            for a in reversed(order):
                v = jax.lax.all_gather(v, a, tiled=True)
            return v * jnp.bfloat16(1.0 / k)

    def f(x):
        return jax.lax.fori_loop(0, INNER, body, x)

    fj = jax.jit(smap(f, mesh, spec, spec))
    x = jax.jit(lambda: jnp.ones((n * per,), jnp.bfloat16), out_shardings=sh)()
    if os.environ.get("C3_HLO") and S == (SIZES_MIB[0] << 20):
        # 编译后 HLO 里实际出现了哪些集合通信指令（看 reduce-scatter 是否被改写成 all-reduce 等）
        import re
        txt = fj.lower(x).compile().as_text()
        cnt = {o: len(re.findall(r"[ =]" + o + r"(?:-start|-done)?\(", txt))
               for o in ("all-reduce", "reduce-scatter", "all-gather", "collective-permute", "all-to-all")}
        cnt = {o: c for o, c in cnt.items() if c}
        emit(kind="hlo", op=op, axes=list(axes) if axes else list(hier_order), collectives=cnt)
    t = timeit(fj, x)
    if op == "hier":  # 输入全 1，正确的分层 all-reduce（再乘 1/k）结果仍全 1
        y = fj(x)
        v = float(jnp.max(jnp.abs(y.astype(jnp.float32) - 1.0)))
        if v > 1e-2:
            raise RuntimeError(f"hier 结果不对：max|y-1|={v}")
    del x
    return t


def main():
    arr, dims, ncore, devs = physical_mesh()
    n = len(devs)
    emit(kind="info", devices=n, processes=jax.process_count(), dims=dims, cores_per_chip=ncore,
         device_kind=devs[0].device_kind, jax=jax.__version__,
         # 一维 make_mesh 的顺序（前 16 个 device 的物理坐标），看它是不是按 torus 走的环
         make_mesh_first16=[list(getattr(d, "coords", [])) + [getattr(d, "core_on_chip", 0)]
                            for d in jax.make_mesh((n,), ("i",)).devices.flat[:16]],
         process0_host_devices=[list(getattr(d, "coords", [])) for d in jax.local_devices()])
    mesh_nd = Mesh(arr, ("x", "y", "z", "c"))
    mesh_1d = jax.make_mesh((n,), ("i",))
    size = dict(zip(("x", "y", "z", "c"), dims + [ncore]))
    # 大轴优先 reduce-scatter（数据最大时走链路最多的维）——另测一种反序做对照
    order_big_first = tuple(sorted(("x", "y", "z"), key=lambda a: -size[a])) + (("c",) if ncore > 1 else ())
    order_c_first = (("c",) if ncore > 1 else ()) + tuple(sorted(("x", "y", "z"), key=lambda a: size[a]))
    cases = []
    for a in ("x", "y", "z", "c"):
        if size[a] > 1:
            for op in ("all_reduce", "all_gather", "reduce_scatter"):
                cases.append((f"axis_{a}", mesh_nd, op, (a,), size[a], None))
    # 轴组合（只测 all-reduce）：看「对称 / 不对称的子网格」各自多快，例如 4x4x8 上 xy=4x4（对称）vs xyz=4x4x8（不对称）
    for combo in (("x", "y"), ("y", "z"), ("x", "y", "z"), ("z", "c")):
        k = 1
        for a in combo: k *= size[a]
        if all(size[a] > 1 for a in combo):
            cases.append(("combo_" + "".join(combo), mesh_nd, "all_reduce", combo, k, None))
    for op in ("all_reduce", "all_gather", "reduce_scatter"):
        cases.append(("all_1d", mesh_1d, op, ("i",), n, None))
        cases.append(("all_nd", mesh_nd, op, ("x", "y", "z", "c"), n, None))
    cases.append(("hier_c_first", mesh_nd, "hier", None, n, order_c_first))
    cases.append(("hier_big_first", mesh_nd, "hier", None, n, order_big_first))
    for mib in SIZES_MIB:
        S = mib << 20
        for label, mesh, op, axes, k, order in cases:
            try:
                t = run_op(mesh, op, axes, S, k, order)
                if t is None:
                    continue
                alg = S / t / 1e9
                fac = 2 * (k - 1) / k if op in ("all_reduce", "hier") else (k - 1) / k
                emit(kind=op, group=label, axes=list(axes) if axes else list(order), k=k, mib=mib,
                     us=round(t * 1e6, 1), algbw_GBps=round(alg, 2), busbw_GBps=round(alg * fac, 2))
            except Exception as e:
                emit(kind=op, group=label, mib=mib, error=repr(e)[:300])


if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    except Exception as e:
        emit(kind="error", error=repr(e)[:500])
    emit(kind="done", seconds=round(time.time() - t0, 1))
