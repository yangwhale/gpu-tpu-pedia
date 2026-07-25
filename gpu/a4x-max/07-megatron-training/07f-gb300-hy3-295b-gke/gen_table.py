#!/usr/bin/env python3
"""从 results.csv 生成 README §十 的消融实验超级大表格。

用法: python3 gen_table.py > /tmp/section10.md
数据源: results.csv（sweep.sh / sweep_d.sh 自动写入）
"""
import csv
import os

D = os.path.dirname(os.path.abspath(__file__))
BASE_TFLOPS = 854.0          # A1 冠军配置（BF16），作为相对增减量的基准
PEAK = {"bf16": 2700.0, "fp8": 5400.0}   # GB300 峰值算力

STATUS_ICON = {"OK": "✅", "OOM": "❌ OOM", "HANG": "⚠️ HANG",
               "CRASH": "❌ CRASH", "SKIP": "⏭ 跳过"}

GROUPS = [
    ("A", "A 组 · 单开关消融",
     "从 A1 冠军配置出发，每次只动一个开关，隔离单项贡献。"),
    ("B", "B 组 · 并行度与批次",
     "追查 854 与 Qwen3-235B 1360 的差距是否来自并行/批次配置。"),
    ("C", "C 组 · 精度",
     "BF16 vs FP8_MX，直接对标 Qwen3 的 MXFP8 口径。"),
    ("D", "D 组 · 规模隔离",
     "层数减半让权重减半，验证「权重挤占显存导致批次开不大」的假设。"),
]


def load():
    rows = list(csv.DictReader(open(os.path.join(D, "results.csv"))))
    # 旧 D 组（idx 22-24）因 40 层配 VPP=8 不能整除而 CRASH，已由补跑替代
    return [r for r in rows
            if not (r["name"].startswith("D") and r["status"] == "CRASH"
                    and r["idx"] in ("22", "23", "24"))]


def cell(r, k, suffix=""):
    v = (r.get(k) or "").strip()
    return f"{v}{suffix}" if v else "—"


def delta(r):
    v = (r.get("tflops_median") or "").strip()
    if not v:
        return "—"
    p = (float(v) - BASE_TFLOPS) / BASE_TFLOPS * 100
    return f"**{p:+.1f}%**" if abs(p) >= 1 else f"{p:+.1f}%"


def main():
    rows = load()
    ok = [r for r in rows if r["status"] == "OK"]
    best = max(ok, key=lambda r: float(r["tflops_median"]))

    print("## 十、消融实验超级大表格（2026-07-26）\n")
    print(f"**{len(rows)} 组配置全量扫点**，由 [`sweep.sh`](sweep.sh) 自动串行执行：")
    print("每组自动清僵尸 CUDA context（清不掉则重建 pod）→ 分发 → 16 pod 启动 → "
          "等稳态 → 采集指标 → 写 [`results.csv`](results.csv)。\n")
    print("**采集口径**：TFLOP/s 取稳态末 5 步中位数（首步含 graph capture，已排除）；"
          "HBM 取全程 `nvidia-smi` 峰值；tok/s/GPU = `GBS × seq_len / step_time / 64`；")
    print("MFU = Model TFLOP/s ÷ 硬件峰值（**BF16 按 2,700，FP8 按 5,400**）。\n")
    print(f"> 基准 **A1 = 854.0 TFLOP/s**（BF16 冠军配置），`vs A1` 列为相对增减量。\n")

    for tag, title, desc in GROUPS:
        sel = [r for r in rows if r["name"].startswith(tag)]
        if not sel:
            continue
        print(f"### {title}\n\n{desc}\n")
        print("| 实验 | 状态 | TFLOP/s | vs A1 | MFU | HBM | tok/s/GPU | Step |")
        print("|---|---|---|---|---|---|---|---|")
        for r in sel:
            name = r["name"].split("_", 1)[1] if "_" in r["name"] else r["name"]
            eid = r["name"].split("_")[0]
            star = " 🏆" if r is best else ""
            print(f"| **{eid}** {name}{star} | {STATUS_ICON[r['status']]} | "
                  f"{cell(r,'tflops_median')} | {delta(r)} | {cell(r,'mfu_pct','%')} | "
                  f"{cell(r,'hbm_gb',' GB')} | {cell(r,'tokens_per_s_per_gpu')} | "
                  f"{cell(r,'step_time_s','s')} |")
        print()

    # ---- 影响力排名 ----
    print("### 单项影响力排名（按实测增益）\n")
    print("| 旋钮 | 增益 | 依据 | 备注 |")
    print("|---|---|---|---|")
    print("| **FP8_MX 精度** | **+50.6%** | A1 854.0 → C1 1285.9 | 最大杠杆，且省 30 GB 显存 |")
    print("| **CUDA graph** | **+44.6%** | A8 572.5 → A9 827.7 | 不开 graph 掉三分之一 |")
    print("| **a2a_overlap** | **+18.8%** | A3 718.6 → A1 854.0 | MoE 通信重叠，唯一显著的融合项 |")
    print("| **FP8 解锁的 MBS=2** | **+5.8%** | C1 1285.9 → C2 1360.4 | BF16 下此路不通 |")
    print("| 层数减半解锁的 MBS=2 | +5.5% | D1 846.4 → D2 892.7 | 与上条同源：腾显存换批次 |")
    print("| EP 32 → 16 | +3.3% | A1 854.0 → B4 882.3 | 需配 PP=2 才兑现 |")
    print("| full_iteration vs TE graph | +3.1% | A9 827.7 → A1 854.0 | 代价 +32 GB |")
    print("| router_fusion | +1.3% | A5 843.1 → A1 854.0 | 边际 |")
    print("| permute_fusion | **0%** | A6 854.2 ≈ A1 854.0 | 被 cutedsl 吸收，纯空转 |")
    print("| 并行度 PP/VPP | **~0%** | B1/B2/B3 全在 852–856 | **调并行不提性能** |\n")

    print("### full_iteration 的硬依赖（缺一即崩）\n")
    print("A2 / A4 / A7 三个 CRASH 共同勾勒出一条**强制依赖链**，不是「可选优化」：\n")
    print("```")
    print("cutedsl_fused_grouped_mlp   ← A2 关掉即 CRASH")
    print("  └→ use_transformer_engine_op_fuser=True      (overrides.py:238)")
    print("       └→ moe_expert_rank_capacity_factor      （固定专家容量）")
    print("            └→ moe_paged_stash                 ← A4 关掉即 CRASH")
    print("                 └→ full_iteration graph 才抓得住 dropless MoE 的变长 tensor")
    print("hybridep dispatcher         ← A7 换 alltoall 即 CRASH")
    print("```\n")

    print("### 显存墙：MBS=2 的四次尝试\n")
    print("| 尝试 | 层数 | 精度 | 手段 | 结果 |")
    print("|---|---|---|---|---|")
    print("| V5 | 80 | BF16 | full graph | ❌ OOM |")
    print("| B5 | 80 | BF16 | 退回 TE graph 省 32 GB | ⚠️ HANG @277 GB |")
    print("| B6 | 80 | BF16 | PP4 摊薄每 stage 激活 | ❌ OOM |")
    print("| **D2** | **40** | BF16 | **层数减半** | ✅ **892.7** |")
    print("| **C2** | 80 | **FP8** | **精度减半权重** | ✅ **1360.4** |\n")
    print("> 80 层 BF16 下 MBS=2 **无解**——换 graph、换并行都救不回来。")
    print("> 只有**减半权重**（减层 or 换 FP8）才能腾出激活空间。这是显存的物理约束，不是调参问题。\n")


if __name__ == "__main__":
    main()
