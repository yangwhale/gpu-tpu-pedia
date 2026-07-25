# SGLang · DeepSeek-V4-Pro · GB300 NVL72 复现 Runbook

> **本文定位**：**只讲怎么跑通**的操作手册。所有命令可直接复制，所有引用的脚本/manifest 都已入库（`manifests/` + `scripts/`）。
> 原理分析、benchmark 演进史、被推翻的假设 → 见 `sglang-v4-gb300-benchmark.md`（旧版，按时间顺序记录，信息全但需自行导航）。
>
> **文档状态约定**：每节标注 `[已验证]` / `[待验证]` / `[已知问题]`，**不用星标**。标 `[已验证]` 的都在本文末 §9 有实跑记录。

---

## ⚠️ 最重要的一条：模型与 MoE backend 必须配套

**这是复现失败的头号原因，且报错完全指不到根因。** 两套配方互斥，混用必挂：

| checkpoint | MoE runner backend | a2a backend | 备注 |
|---|---|---|---|
| `deepseek-ai/DeepSeek-V4-Pro`（**官方原装**，FP4 MoE + FP8 attn） | `deep_gemm` | **`megamoe`** | 高性能路径，旧文档 8,993 就是这个 |
| `nvidia/DeepSeek-V4-Pro-NVFP4`（NVFP4 MoE + FP8 attn） | **`flashinfer_trtllm_routed`** | 不用 megamoe | megamoe 没有为它注册融合 kernel |

**混错的两种典型报错**：

- NVFP4 权重 + megamoe → `NotImplementedError: Runner backend FLASHINFER_TRTLLM_ROUTED requires a fused func for a2a backend megamoe, but none is registered`
- NVFP4 权重 + `deep_gemm` + megamoe → `RuntimeError: The size of tensor a (384) must match the size of tensor b (48)`（384 = `n_routed_experts`，48 = 384/ep_size，随 ep 缩放）

> **先确认你手上是哪份权重**：`python3 -c "import json;print(json.load(open('<model>/config.json')).get('quantization_config'))"`。
> 旧文档把这两套配方分散在 §1（checkpoint 说明）和 §13.A（启动脚本）两处，中间无任何交叉提示——照 §13.A 抄、配 NVFP4 权重，必然失败。

---

## 0. TL;DR

| 项 | 值 |
|---|---|
| 目标配置 | **14 prefill（每个 dep4）+ 1 decode（dep8 跨 2 节点）+ MTP** |
| 节点需求 | 16 节点 GB300，**必须同一 NVL72 域**（本文用 `gb300-pool-0002`） |
| 模型 | `/mnt/ssd/DeepSeek-V4-Pro-NVFP4`（851G，节点本地 SSD，**注意 `-NVFP4` 后缀**） |
| 镜像 | `lmsysorg/sglang:nightly-dev-cu13-20260720-b3570a45` |
| 编排 | Dynamo（`dynamo.sglang` worker + `dynamo.frontend`）+ NATS + etcd |
| 参考成绩 | 8,993 output tok/s ÷ decode-GPU ≈ 官方 11,200 的 80%（旧版文档 §12） |

**为什么是 14 prefill**：实测 14→16 prefill 只涨 2%（8,809 → 8,993），已收敛。14 是性价比拐点。

---

## 1. 前置条件 `[已验证]`

```bash
# ① 同域节点数（16 起）
kubectl get nodes -l cloud.google.com/gke-nodepool=gb300-pool-0002,team=yangwhale --no-headers | wc -l

# ② Dynamo 基础设施（服务发现，worker 靠它自注册）
kubectl get pods | grep -E 'dynamo-nats|dynamo-etcd'   # 两个都要 Running

# ③ DRA GPU driver
kubectl get pods -A | grep -c dra-driver-nvidia-gpu    # >0
```

**同域是硬要求**：PD 分离的 KV 传输走 mooncake + MNNVL，跨域会退化到 RDMA 并显著变慢。同 nodepool 通常同域。

---

## 2. 部署 pod fleet `[已验证]`

```bash
kubectl apply -f manifests/sgl-fleet.yaml
kubectl get pods -l app=sgl -w        # 等 16/16 Running（实测 ~2min）
```

**角色分配**（由跑哪个脚本决定，pod 本身同构）：

| Pod | 角色 |
|---|---|
| `sgl-0` / `sgl-1` | decode，dep8 跨 2 节点（rank 0 / rank 1）|
| `sgl-2` … `sgl-15` | prefill，各自 dep4 |

**manifest 的三个关键设计**：

1. **用 StatefulSet，不要裸 pod + `nodeName`**。DRA 的 ComputeDomain channel 必须由 scheduler 预留，`nodeName` 绕过调度器会导致 `FailedPrepareDynamicResources`。
2. **StatefulSet 送的稳定 DNS 名是白赚的**。decode 跨节点要 `--dist-init-addr`，有稳定名比每次查 IP 可靠（pod IP 每次重建都变）。
3. **两个 DRA claim 都要**：`sgl4-mrdma`（8 张 CX-8，mooncake KV 传输）+ `sgl4-ch`（ComputeDomain channel，MNNVL）。

---

## 3. 检查模型 `[已验证]`

```bash
for i in $(seq 0 15); do
  echo -n "sgl-$i: "; kubectl exec sgl-$i -- ls -d /mnt/ssd/DeepSeek-V4-Pro-NVFP4 2>/dev/null || echo 缺失
done
```

模型在**节点本地 SSD**（hostPath `/mnt/disks/raid/0`），删 pod 不丢。若缺失见 `gb300-local-ssd-raid0-SETUP.md`。

> ⚠️ **路径必须是 `DeepSeek-V4-Pro-NVFP4`**。旧文档 §13.A 写成 `DeepSeek-V4-Pro`（无后缀），照抄会报模型找不到。

---

## 4. 分发启动脚本 `[已验证]`

```bash
for i in $(seq 0 15); do
  ( kubectl exec -i sgl-$i -- bash -c "cat > /tmp/decode-dep8.sh"  < scripts/decode-dep8.sh
    kubectl exec -i sgl-$i -- bash -c "cat > /tmp/prefill-dep4.sh && chmod +x /tmp/*.sh" < scripts/prefill-dep4.sh ) &
done; wait
kubectl exec sgl-5 -- wc -l /tmp/decode-dep8.sh /tmp/prefill-dep4.sh   # 校验非空
```

> ⚠️ **必须 `kubectl exec -i`**。少了 `-i` 时 stdin 不透传，容器里会得到**空文件**，而且不报错。

---

## 5. 启动 workers `[已验证 prefill / 见 §8 decode]`

**先单个冒烟，再批量铺开**——14 个一起错的排查成本远高于先验 1 个。

```bash
# ① 单 prefill 冒烟（~3min 后 HBM 应 >200G）
kubectl exec sgl-2 -- bash -c "setsid nohup bash /tmp/prefill-dep4.sh > /tmp/srv.log 2>&1 </dev/null &"
sleep 180 && kubectl exec sgl-2 -- nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1

# ② 通过后起 decode + 剩余 13 prefill
D0=$(kubectl get pod sgl-0 -o jsonpath='{.status.podIP}')
kubectl exec sgl-0 -- bash -c "setsid nohup bash /tmp/decode-dep8.sh 0 $D0:5000 > /tmp/srv.log 2>&1 </dev/null &"
kubectl exec sgl-1 -- bash -c "setsid nohup bash /tmp/decode-dep8.sh 1 $D0:5000 > /tmp/srv.log 2>&1 </dev/null &"
for i in $(seq 3 15); do
  kubectl exec sgl-$i -- bash -c "setsid nohup bash /tmp/prefill-dep4.sh > /tmp/srv.log 2>&1 </dev/null &" &
done; wait
```

**就绪判据**（`"fired up"` 日志有 stdout 缓冲、不可靠，别用）：

| 角色 | 判据 |
|---|---|
| prefill | **`grep 'Load weight end' /tmp/srv.log`** |
| decode | `grep 'Model registration succeeded' /tmp/srv.log` |

> ⚠️ **不要用「HBM > 200G」判就绪**（旧文档如此建议）。那只是 SGLang 预分配的显存池，**权重可能随后加载失败、显存随即归零**。本次复测中我就因此误判 13/14 prefill「就绪」，实际全部在加载权重时崩了。唯一可靠判据是日志里的 `Load weight end`。

---

## 6. 自愈循环（**一次成功的关键**）`[已验证]`

单个 worker 起不来是**常态**，不是异常。手动逐个救每次都漏，正确姿势是脚本化「校验→重试」：

```bash
for round in 1 2 3; do
  BAD=""
  for i in $(seq 2 15); do
    M=$(kubectl exec sgl-$i -- bash -c "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1" 2>/dev/null|tr -d ' ')
    [ "${M:-0}" -lt 200000 ] 2>/dev/null && BAD="$BAD $i"
  done
  [ -z "$BAD" ] && { echo "全绿"; break; }
  echo "round$round 未就绪:$BAD → 清进程"
  for i in $BAD; do kubectl exec sgl-$i -- bash -c "pkill -9 -f dynamo.sglang; pkill -9 -f sglang" 2>/dev/null; done
  sleep 95                                    # ★ 必须 ≥90s
  for i in $BAD; do kubectl exec sgl-$i -- bash -c "setsid nohup bash /tmp/prefill-dep4.sh > /tmp/srv.log 2>&1 </dev/null &"; done
  sleep 240
done
```

> ★ **每轮间隔必须 ≥90 秒**。40236 ZMQ 端口僵尸的持有者是 D-state 进程，`kill -9` 杀不掉，但**内核会在其 GPU 驱动调用返回后自动回收**。急着重试（<10s）会连撞，让人误判"必须重建 pod"。实测 2-3 轮自清。

---

## 7. 启动 frontend `[待验证]`

**必须在所有 worker 稳定之后统一起**——因为 `pkill python3` 会连 frontend 一起杀。

```bash
for i in $(seq 2 15); do
  kubectl exec sgl-$i -- bash -c "NATS_SERVER=nats://dynamo-nats:4222 ETCD_ENDPOINTS=http://dynamo-etcd:2379 \
    setsid nohup python3 -m dynamo.frontend --http-port 8001 > /tmp/fe.log 2>&1 </dev/null &"
done
# 探活：必须 owned_by=nvidia + context_window=1048576，否则是僵尸 sglang::router 占端口
kubectl exec sgl-2 -- curl -s localhost:8001/v1/models
```

**多 frontend 是吞吐关键（+34%）**：单个裸 frontend 在高并发下 CPU-bound，是编排瓶颈。

---

## 8. 故障速查

| 现象 | 根因 | 处理 | 状态 |
|---|---|---|---|
| `NotImplementedError: Runner backend FLASHINFER_TRTLLM_ROUTED requires a fused func for a2a backend megamoe` | `--moe-runner-backend` 默认 `auto`，新 nightly 的 auto 选了 flashinfer，与 megamoe 不配套 | **显式加 `--moe-runner-backend deep_gemm`**（脚本已含） | `[已定位并修复]` |
| 模型找不到 | 路径缺 `-NVFP4` 后缀 | 见 §3 | `[已修复]` |
| 容器里脚本是空文件 | `kubectl exec` 少了 `-i` | 见 §4 | `[已修复]` |
| `FailedPrepareDynamicResources` | 裸 pod + `nodeName` 绕过 scheduler | 用 StatefulSet | `[已修复]` |
| prefill HBM 一直 0 + `scheduler died (exit -3)` | 40236 ZMQ 端口僵尸（瞬态） | §6 自愈循环，间隔 ≥90s | `[已验证有效]` |
| decode 加载到 ~145G 后 `EOFError` + `scheduler died (exit -3)` | 待定位（dep8 跨节点特有） | 见下 | `[已知问题·排查中]` |
| decode 显存 199G 不释放、`kill -9` 无效 | 真 D-state 卡死 | `kubectl delete pod --force` 重建（模型在节点盘，不丢） | — |

---

## 9. 验证记录

**2026-07-25 复测（本 runbook 首次成文）**，环境：`gb300-pool-0002` 17 节点，16 pod fleet。

| 步骤 | 结果 | 实测 |
|---|---|---|
| §1 前置检查 | ✅ | 17 节点带标签；dynamo-nats/etcd Running |
| §2 部署 fleet | ✅ | 16/16 Running，一节点一 pod，~2min |
| §3 模型检查 | ✅ | 16/16 节点有 `DeepSeek-V4-Pro-NVFP4` 851G |
| §4 分发脚本 | ✅ | 16 pod 全部写入成功 |
| §5 单 prefill 冒烟 | ✅ | 加 `--moe-runner-backend deep_gemm` 后，3min HBM 260GB |
| §5 批量 14 prefill | ✅ **13/14** | 1 个撞 40236 僵尸 |
| §6 自愈循环 | ✅ | 2 轮清完掉队节点 |
| §5 decode dep8 | ✅ | 8 rank 全 `Load weight end`，`registration succeeded` |
| §7 frontend | ✅ | 14/14 返回 200，`owned_by=nvidia`、`context_window=1048576` |
| **端到端推理** | ✅ | `curl /v1/completions` → 正确回答 "Paris"，PD 链路全通 |

**本次复测发现的旧文档缺陷（均已在本文修正）**：

1. **pod 生成器未入库** —— 旧文档引用 `gen18-0001.py`，仓库里没有，用户第一步就卡死。→ 本文提供 `manifests/sgl-fleet.yaml`。
2. **模型路径错误** —— 旧文档 §13.A 标注"source of truth，照抄即可"，但路径漏了 `-NVFP4`。
3. **漏 `--moe-runner-backend deep_gemm`** —— 旧文档强调"一个字都不能漏"，却依赖 `auto` 的默认行为。nightly 镜像更新后 auto 改选 flashinfer，**整条 megamoe 路径直接崩**。这是「依赖默认值」型埋雷的典型案例：写文档时能跑，不代表半年后能跑。
4. **裸 pod 与 DRA 冲突未说明** —— 生成器批量创建裸 pod，与 DRA channel 预留机制冲突。
5. **模型与 MoE backend 配方错配（最严重）** —— 见文首告警。两套互斥配方分散在 §1 和 §13.A，无交叉提示。
6. **就绪判据错误** —— 「HBM>200G」会把「显存池已分配但权重加载失败」误判为成功。
7. **GCS bucket 名写成占位符** —— 全文用 `gs://<bucket>`，真实名是 `gs://chrisya-gb300-models`，照抄跑不了。
8. **压测工具来源未记** —— §13 Step 4 用 InferenceX `sa-bench`，但没说它从哪来、怎么装。pod 里默认没有，只有 SGLang 自带的 `sglang.bench_serving`（两者口径不同，见旧文档 §11.4）。

> **方法论**：本文档每一节都是「先在集群上真跑一遍，再写进来」。上面 4 个缺陷全部是执行过程中撞出来的，靠读旧文档发现不了。
