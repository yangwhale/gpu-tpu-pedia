> 🌐 [中文](QUICKSTART-v7.md) | **English**
>
> 🚧 **Translation in progress.** Links marked 🇨🇳 still point to the Chinese version; their English counterparts are being added document by document.

# Pre-training Hunyuan 3 (295B-A21B) on TPU v7 (Ironwood) — Quick Start

Running Tencent Hunyuan 3 on TPU v7 with MaxText. **This document hands you the best recipe directly — follow it and you get today's highest water line on the first try, with no tuning of your own.**

| | |
|---|---|
| Model | Tencent Hunyuan 3, 295B total / 21B active, **full 80 layers**, MoE |
| Platform | TPU v7 Ironwood (`tpu7x-standard-4t`, **2 devices / chip**) |
| Framework | MaxText (nnx); code lives on the [`hunyuan3` branch of `yangwhale/maxtext`](https://github.com/yangwhale/maxtext/tree/hunyuan3) |
| Precision | BF16 compute / FP32 master weights |

**Measured water lines at two scales** (2026-08-04; all full 80 layers, seq 4096, synthetic data, steady state taken from steps 4–7):

| Scale | Parallelism | pdbs | step | **TFLOP/s/chip** | **MFU** | **tok/s** | tok/s/chip | Peak HBM |
|---|---|---|---|---|---|---|---|---|
| **256 chips**, maximum | `DP=2 × FSDP=256` | **16** | 30.40 s | **599** | **25.96%** | **1,103,757** | **4,312** | 92.33 G |
| **256 chips**, recommended | `DP=4 × FSDP=128` | 12 | 23.56 s | 580 | 25.12% | 1,068,372 | 4,173 | 91.94 G |
| **64 chips** (16 nodes / 128 dev) | `DP=1 × FSDP=128` | 12 | 23.54 s | 580 | 25.14% | 267,284 | 4,176 | 91.94 G |
| Reference: untuned starting point (64 chips) | `FSDP=128` | 8 | 19.92 s | 457 | 19.80% | 210,570 | 3,290 | 74.20 G |
| 🔬 **FP8 (no QAG)** | `DP=2 × FSDP=256` | 16 | 29.46 s | 618 | 13.39%<sub>against the FP8 peak of 4614</sub> | 1,139,022 | 4,449 | 92.80 G |
| ⭐ **FP8 + QAG** (best at 64c) | `DP=2 × FSDP=64` | 7 | 12.73 s | **625** | 13.55%<sub>against the FP8 peak</sub> | 288,222 | **4,503** | 92.42 G |

> **The MFU denominator for FP8 is 4614, not BF16's 2307 — do not compare it head-to-head with the rows above.**
> With QAG on at 64 chips the figure is **625 / 13.55%**; the official DSV3 number under the same convention is 743.5 / 16.1%.
>
> ⚠️ **This document previously claimed that "the tile on that FP8 kernel has never been swept, potential ~726." That claim has been disproven by measurement.**
> Under `use_tokamax_gmm=True`, FP8 **still routes back into tokamax internally**, so the tile monkeypatch has been in effect all along;
> on 2026-08-05 we additionally swept tile / XLA flags / SparseCore offloading / larger batch, 8 cells in total, with **not a single positive result**.
> Closing the remaining gap requires more chips, a different model shape, or writing code — it is outside the tuning space.
> See [TUNING-v7 §4.6, the master table](TUNING-v7.en.md#46-what-can-be-tuned-and-what-cannot--one-master-table).

> **Token throughput = `device count × pdbs × seq ÷ step`.** For cross-platform comparison look only at **tok/s/chip**;
> cluster tok/s scales with the number of units and is not comparable. Two reference points:
> **GB300 = 6,242 tok/s/GPU** (also seq 4096, so **directly comparable**) — v7 is now at **69.1%** of its per-GPU figure, up from 51.4% before tuning;
> **v5p = 1,037 tok/s/chip** (⚠️ v5p used **seq 8192**, a different convention, so any multiple computed against it is inflated; treat it as an order-of-magnitude reference only).

> ### 🔬 64 chips and 256 chips landed on the same number
>
> **580 vs 580, MFU 25.14% vs 25.12%, peak HBM 91.94 G vs 91.94 G — identical to the byte.**
>
> This is not a coincidence: `DP=4 × FSDP=128` is simply **four independent 64-chip jobs**.
> Within a group they do FSDP collectives; across groups they synchronize gradients exactly once, at the end of each step.
> The per-device shard shapes, tile matching, and activation footprint are all identical, so per-chip performance must be identical too.
>
> ⇒ **You do not need to grab a large slice for performance; 16 nodes already give you the full gain.**
> ⇒ **Tuning can be done on 16 nodes**, as long as the knobs you change do not alter the shard shapes (neither tile nor pdbs may change).
>
> **So where does the extra 3.3% at 256 chips (599) come from?** Not from scale, but from **having an additional `FSDP=256` option** —
> doubling FSDP halves the per-device static shard, freeing 13 G of HBM, which is enough to push pdbs from 12 all the way to 16.
> 64 chips only has 128 devices, so **there is no wider FSDP available, which makes 580 its ceiling.**
>
> The target is 600–630 (the realistic water line for sparse MoE on Ironwood, see [TUNING-v7 §3](TUNING-v7.en.md#12-why-the-target-is-600630-not-900)).
> We are at 599, **0.2% short**. For why that is the target and what remains untried, see
> **[TUNING-v7.md — performance tuning in practice](TUNING-v7.en.md)**.

---

## 0. Best recipe, at a glance

**Three things account for the entire climb from 457 to 580**, ordered by contribution:

| # | Lever | What it's worth | Cost |
|---|---|---|---|
| 1 | **tokamax `tile(512, 2048, 1536)`** | **+17.4%** | Requires a 6-line monkeypatch ([§3.3](#33-inject-the-tokamax-tile-mandatory-the-single-largest-gain)); HBM +1.1 G |
| 2 | **`per_device_batch_size` 8 → 12** | **+9.0%** | HBM 74 → 92 G, right up against the 94.74 limit |
| 3 | **Choosing the right `DP × FSDP` split** (FSDP fixed at 128) | see below | none |

**What item 3 means**: 512 devices can be split several ways, but the usable range is narrow —

```
Splits for 512 devices (compared at a fixed pdbs of 8):
FSDP=512 (DP=1)  → 404  ❌ spread too thin; collective shards fragment, costing 11%
FSDP=256 (DP=2)  → 450  ⭕ ties with 128, but saves 13 G of HBM ← those 13 G matter later
FSDP=128 (DP=4)  → 453  ✅
FSDP=64  (DP=8)  → OOM  ❌ per-device static shard doubles, blowing out HBM
FSDP=32  (DP=16) → OOM  ❌
```

**Default rule: keep FSDP width fixed at 128 and give every additional device to DP.**
64 chips is exactly 128 devices, hence `DP=1`; 256 chips is 512 devices, hence `DP=4`; 1024 devices would be `DP=8`.

**But at ≥ 256 chips there is one more move available: widen FSDP to 256 and spend the 13 G you save on a larger batch.**

| Two routes for 512 devices | FSDP | pdbs | chip | HBM | Trade-off |
|---|---|---|---|---|---|
| **Recommended** (isomorphic to 64 chips) | 128 | 12 | 580 | 91.94 G | The recipe is identical to the 64-chip one, so the pattern extrapolates |
| **Maximum** (squeeze the HBM) | 256 | **16** | **599** | 92.33 G | **3.3% higher**, but the recipe does not transfer to smaller scales |

**Use DP=4 if you want portability, DP=2 + pdbs 16 if you want the peak number.**
The latter is 3.3% higher (beyond the ±3% noise band, so it is a real difference), at the cost of a recipe that cannot be reproduced on 64 chips.

**Do not use EP (expert parallelism).** TPU's ICI is a 3D torus, so AllToAll requires multi-hop forwarding —
unlike the full mesh of GPU NVLink. Measured on 16 chips, EP=4 costs **−71%**
([TUNING-v7 §7.4](TUNING-v7.en.md#37-how-to-split-parallelism-the-usable-range-is-only-fsdp--128-256)), and there is no physical reason for that to flip sign at larger scale.

### The two parameter sets you can copy directly

**64 chips (16 nodes / 128 devices) → 580**

```
ici_fsdp_parallelism=-1      # auto-fills all 128 ways, equivalent to DP=1
ici_tensor_parallelism=1
per_device_batch_size=12     # 91.94 G, right at the limit; 14 will OOM
megablox=True use_tokamax_gmm=True
TK_TM=512 TK_TK=2048 TK_TN=1536      # environment variables, paired with tkcfg.py
```

**256 chips (64 nodes / 512 devices) → 580, isomorphic to the 64-chip case**

```
ici_data_parallelism=4       # ← the only difference from the 64-chip version
ici_fsdp_parallelism=128
ici_tensor_parallelism=1
per_device_batch_size=12
megablox=True use_tokamax_gmm=True
TK_TM=512 TK_TK=2048 TK_TN=1536
```

**256 chips, maximum version → 599** (the recipe does not transfer to smaller scales; usable only at ≥ 256 chips)

```
ici_data_parallelism=2
ici_fsdp_parallelism=256     # widen FSDP: per-device static shard halves, saving 13 G
ici_tensor_parallelism=1
per_device_batch_size=16     # feed every byte saved to the batch, 92.33 G
megablox=True use_tokamax_gmm=True
TK_TM=512 TK_TK=2048 TK_TN=1536
```

All other parameters are identical across the three; see [§4.3](#43-the-complete-parameter-set).

---

## 1. Model and code

### 1.1 What Hy3 is

In one sentence: **the attention is Qwen3's, the MoE is DeepSeek V3's**.
Both halves already have working implementations inside MaxText, so this project wrote only the assembly logic — zero new math.

| | |
|---|---|
| Structure | 80 layers; layer 0 dense, layers 1–79 MoE |
| Attention | GQA 64q / 8kv, head_dim 128, QK-LayerNorm, no bias |
| MoE | **192** routed experts top-8 plus 1 shared, sigmoid routing with expert bias |
| Other | 1 MTP layer, vocab 120832, routed scaling 2.826 |
| Parameter distribution | **97% sits in the routed experts**; attention accounts for only 2% |

That parameter distribution dictates the parallelism strategy directly: **TP is useless** (sharding attention is pure communication loss), **EP is a negative optimization**, and **FSDP does the heavy lifting**.

> ⚠️ **192 is not a power of two, and that number will bite you repeatedly.**
> `shard_exp_on_fsdp=True` raises `IndivisibleError` outright on 128 devices (192 % 128 ≠ 0);
> EP can likewise only take divisors of 192. Check divisibility before picking a parallelism.

> The full architectural breakdown — why neither existing decoder block works, the key difference from DSV3
> (Hy3 has no device-limited routing), and the parameter-count decomposition — is in
> [v5p Quick Start §1](QUICKSTART-v5p.md) 🇨🇳. **It is identical on both platforms and is not repeated here.**

### 1.2 Where the code comes from

**The fork branch is the single source of truth**; no copy of the code is kept in this repo:

```
https://github.com/yangwhale/maxtext   branch hunyuan3
```

Based on upstream main, three commits: the model itself plus registration; the upstream loss-free-balancing bias-path fix
(unrelated to Hy3 — any non-DeepSeek model that enables aux-loss-free balancing will hit it); and the SwiGLU activation-bound allowlist.

**3 new files** (`models/hunyuan3.py`, 161 lines of effective code, plus two yml files);
the 12 modified upstream files are all about "making the framework recognize this model" — not one of them implements an algorithm.

**v5p and v7 use the same code and the same image**; only the launch parameters differ.

---

## 2. Environment setup

> **Using someone else's managed cluster (Kueue / queue-based)? Skip §2.1 and §2.2** — that kind of cluster will not let you create
> workload policies or node pools yourself; you submit a job and it hands you machines. But read [§3.7](#37-diagnosing-a-managed-kueue-cluster-that-wont-scale-up) first.

### 2.1 Workload policy (v7-specific, must be created first)

On v5p, creating a node pool takes a single command. tpu7x rejects it outright:

```
Creation of a managed instance group with tpu7x-standard-4t machine type
with placement policy is not supported. Use workload policy instead.
```

**Use the gcloud subcommand directly** (re-verified 2026-08-07: it exists and works on `gcloud 577.0.0`):

```bash
P=YOUR-PROJECT
for TOPO in 4x4x4 4x8x8 2x2x1; do
  gcloud compute resource-policies create workload-policy wp-$TOPO \
    --project=$P --region=us-central1 \
    --type=HIGH_THROUGHPUT --accelerator-topology=$TOPO
done
```

- `--accelerator-topology` **is mandatory**. Passing only `--type` produces
  `does not support TPU topology with group placement policy and workload policy at the same time`
- **One policy per topology** — create as many as the number of topologies you use

<details><summary>REST fallback, for a gcloud that lacks the subcommand</summary>

```bash
TOK=$(gcloud auth application-default print-access-token)
curl -s -X POST -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" \
 "https://compute.googleapis.com/compute/v1/projects/$P/regions/us-central1/resourcePolicies" \
 -d "{\"name\":\"wp-$TOPO\",\"workloadPolicy\":{\"type\":\"HIGH_THROUGHPUT\",\"acceleratorTopology\":\"$TOPO\"}}"
```

</details>

### 2.2 TPU node pool

```bash
# 64 chips (16 machines) — topology 4x4x4
gcloud container node-pools create np-v7x-64 \
  --cluster=CLUSTER --project=$P --region=us-central1 --node-locations=us-central1-c \
  --machine-type=tpu7x-standard-4t --tpu-topology=4x4x4 \
  --placement-policy=wp-4x4x4 --num-nodes=16 \
  --disk-type=hyperdisk-balanced --disk-size=200 --scopes=cloud-platform

# 256 chips (64 machines) — topology 4x8x8, everything else the same
#   --tpu-topology=4x8x8 --placement-policy=wp-4x8x8 --num-nodes=64
```

Four things that differ from v5p:

| | Notes |
|---|---|
| `--placement-policy` | Points at the one created in §2.1; **the topology must match** |
| `--disk-type=hyperdisk-balanced` | **v7 does not accept plain pd** |
| zone | v7 is in **`us-central1-c`**, v5p in `-a` |
| `--num-nodes` | = chip count ÷ 4, and must be consistent with the product of `--tpu-topology` |

> To hold capacity for a long stretch without being preempted, use **DWS flex-start**. Three parameters are locked together:
> `--flex-start` + `--num-nodes=0` + `--enable-autoscaling --min-nodes=0 --max-nodes=N`.
> Omit any one and you get its own error (`Flex start node pools require autoscaling enabled.` /
> `... require initial node count to be set to 0.`). Use `--max-nodes` for the cap —
> `--total-max-nodes` is read as 0.
>
> ⚠️ **Do not add `--enable-queued-provisioning` reflexively.** From the gcloud help:
> *"all new nodes can be obtained **only** through queuing via ProvisioningRequest API"* —
> once set, an ordinary Job or Deployment can no longer bring nodes up; you must install Kueue and submit a `ProvisioningRequest`.
> Only on that path does it need to be paired with `--flex-start` (and there, supplying only queued-provisioning
> yields the misleading `Queued_provisioning doesn't support TPUs`, which actually means `--flex-start` is missing).
> **If you want `sleep infinity` to trigger the scale-up directly, use `--flex-start` alone.**
>
> Expect the queue to take on the order of **20 hours**, not minutes.

### 2.3 JobSet CRD

```bash
kubectl apply --server-side -f \
  https://github.com/kubernetes-sigs/jobset/releases/download/v0.11.1/manifests.yaml
kubectl wait --for=condition=Available deploy/jobset-controller-manager \
  -n jobset-system --timeout=180s
```

v0.11.1 ships its own certificates; **cert-manager is not needed**.

### 2.4 Staging bucket and cross-project permissions

```bash
gcloud storage buckets create gs://YOUR-STAGE-BUCKET --location=US

NODE_SA=<cluster project number>-compute@developer.gserviceaccount.com
gcloud storage buckets add-iam-policy-binding gs://YOUR-STAGE-BUCKET \
  --member="serviceAccount:$NODE_SA" --role=roles/storage.objectViewer

# If the image lives in another project, the node SA also needs pull access
gcloud artifacts repositories add-iam-policy-binding gcr.io --location=us \
  --project=IMAGE_PROJECT --member="serviceAccount:$NODE_SA" \
  --role=roles/artifactregistry.reader
```

### 2.5 Network

The default VPC of a shared project is in auto mode, `10.128.0.0/9` is fully taken by per-region subnets,
and GKE cannot assemble a contiguous `/14` for pods. **Creating your own custom VPC is by far the easiest path**, and it lets you raise the MTU to 8896 along the way:

```bash
gcloud compute networks create NAME-vpc --subnet-mode=custom --mtu=8896
gcloud compute networks subnets create NAME-uc1 --network=NAME-vpc \
  --region=us-central1 --range=10.124.0.0/22 \
  --secondary-range=pods=10.125.0.0/16,services=10.124.16.0/20 \
  --enable-private-ip-google-access
```

### 2.6 Check capacity before creating the pool

**Quota decides whether you may ask; capacity decides whether you get it. These are two independent gates.**
If the pool sits at `PROVISIONING` with no MIG errors, you simply cannot get machines — **switching projects or raising quota will not help**.

```bash
gcloud compute instances list --project=ANY-PROJECT-IN-SAME-ZONE \
  --filter="machineType~tpu7x AND status=RUNNING" \
  --format='value(zone,scheduling.provisioningModel)' | sort | uniq -c
```

> `4x4x4` (64 chips) is an **atomic slice**; there is no intermediate 48 or 56 tier — **not filling it means getting nothing**.
> Only 4 zones worldwide carry the `tpu7x-standard-4t` machine type; in the other zones it physically does not exist, which is not a quota problem.

---

## 3. Running it

### 3.1 A long-lived environment beats one-shot Jobs

**Strongly recommended: use long-lived pods running `sleep infinity` rather than submitting a one-shot Job per round.** Three reasons:

1. **You keep hold of the slice** — on a shared cluster, letting go means someone else takes it; we measured a slice being grabbed within 30 seconds of release
2. **You reuse the compilation cache** — `jax_cache_dir` lives inside the pod, so from the second round on, compilation drops from 10+ minutes to seconds,
   bringing a full round down to about **6 minutes**
3. The code only has to be pulled once

```yaml
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata: {name: hy3-dev, namespace: <NS>}
spec:
  failurePolicy: {maxRestarts: 3}          # ⚠️ do not set 10, see §3.6
  replicatedJobs:
  - name: slice-job
    replicas: 1
    template:
      spec:
        parallelism: 64                     # write 16 for the 64-chip version
        completions: 64
        backoffLimit: 0
        template:
          spec:
            restartPolicy: Never
            nodeSelector:
              cloud.google.com/gke-tpu-accelerator: tpu7x
              cloud.google.com/gke-tpu-topology: "4x8x8"     # write 4x4x4 for the 64-chip version
              # on managed clusters you usually also need reservation-name / queue-name / priorityClassName
            hostNetwork: true
            dnsPolicy: ClusterFirstWithHostNet
            tolerations: [{operator: "Exists"}]
            containers:
            - name: jax-tpu
              image: <MAXTEXT_RUNNER_IMAGE>
              securityContext: {privileged: true}
              ports: [{containerPort: 8471}, {containerPort: 8080}]
              command: ["bash","-c"]
              args: ["gcloud storage cp <GCS_STAGE>/hy3-maxtext.tgz /tmp/p.tgz &&
                      cd /deps && rm -rf src/maxtext && tar xzf /tmp/p.tgz && sleep infinity"]
              resources: {limits: {google.com/tpu: 4}}
              volumeMounts: [{mountPath: /dev/shm, name: dshm}]
            volumes: [{name: dshm, emptyDir: {medium: Memory}}]
```

The code bundle is produced by `prep.sh` (clone the `hunyuan3` branch → 8 self-checks → tar the whole `src/maxtext` tree → upload to GCS).
**Note that it overwrites the entire tree rather than injecting only the changed files** — with injection alone you would be testing "my changes on top of the container's stale base."

### 3.2 Lock-step execution: on multi-host TPU every pod must start together

```bash
#!/bin/bash
# hy3-run.sh —— run the same command in parallel across every pod
CMD=${1:?}; NS=${NS:-default}; JS=${JS:-hy3-dev}; NP=${NP:-64}
mapfile -t PODS < <(kubectl get pods -n $NS -l jobset.sigs.k8s.io/jobset-name=$JS \
  --field-selector status.phase=Running --no-headers | awk '{print $1}' | sort)
[ ${#PODS[@]} -eq $NP ] || { echo "need $NP Running pods, currently ${#PODS[@]}"; exit 1; }
echo "[hy3-run] running in parallel across ${#PODS[@]} pods (live output below is worker-0's)"
TMP=$(mktemp -d); trap 'rm -rf $TMP' EXIT
for i in "${!PODS[@]}"; do
  if [ "$i" -eq 0 ]; then
    # ⚠️ worker-0 must be tee'd live, otherwise you see nothing at all for the 6–30 minutes
    #    a round takes, and cannot tell "compiling" from "hung"
    timeout -k 30 2700 kubectl exec "${PODS[$i]}" -n $NS -c jax-tpu -- \
      bash -c "$CMD" 2>&1 | tee "$TMP/0.out" &
  else
    timeout -k 30 2700 kubectl exec "${PODS[$i]}" -n $NS -c jax-tpu -- \
      bash -c "$CMD" > "$TMP/$i.out" 2>&1 &
  fi
done
wait
grep -lE "^Traceback" $TMP/*.out | sed 's/^/⚠ error: /'
grep -ohE "SLICE_FAILURE_[A-Z_]+" $TMP/*.out | sort -u | sed 's/^/🔴 hardware fault: /'
```

> **How to tell alive from dead when there is no output**: `kubectl exec <pod> -- ps -eo stat,pcpu,etime,comm --sort=-pcpu | head -3`.
> A `%CPU` in the hundreds means multi-threaded compilation (normal); near zero with no `train.py` is a genuine hang.

**The smoke test comes first, always**:

```bash
NS=<ns> JS=<jobset> NP=<pod count> bash hy3-run.sh 'python3 -c "import jax;print(jax.device_count())"'
```

| Scale | `NP` | Should return |
|---|---|---|
| 16 nodes / 64 chips | 16 | **128** |
| 64 nodes / 256 chips | 64 | **512** |

If the number is wrong, do not proceed — one missing pod on a multi-host TPU makes every subsequent experiment worthless.

> ⚠️ **Never `kubectl exec` into a single pod and `import jax` there on its own.**
> That process grabs `/dev/vfio/*` and will not let go; every training run afterwards reports
> `Device or resource busy; Couldn't open iommu group`, and the only fix is to recreate the pod.

### 3.3 Inject the tokamax tile (mandatory, the single largest gain)

MaxText does not expose tokamax's tile parameters, and **the default falls back to `128³`, which is 12.4× slower**.
Inject them with a 6-line monkeypatch:

```python
# tkcfg.py —— exec this before importing train
import os, dataclasses
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu as P
_TM, _TK, _TN = (int(os.environ[k]) for k in ("TK_TM", "TK_TK", "TK_TN"))
_orig = P.PallasMosaicTpuRaggedDot._get_heuristics_config
def _patched(self, ba):
    c = _orig(self, ba)
    k, n = ba.arguments["rhs"].shape[-2], ba.arguments["rhs"].shape[-1]
    return dataclasses.replace(c, tile_m=_TM, tile_k=min(_TK, k), tile_n=min(_TN, n))
P.PallasMosaicTpuRaggedDot._get_heuristics_config = _patched
print("[tkcfg] patched")
```

`kubectl cp` it to `/tmp/tkcfg.py` in every pod (do the copies in parallel; 64 of them in series is painfully slow).

**Why `(512, 2048, 1536)`** — three empirical rules:

| Dimension | Optimum | Notes |
|---|---|---|
| `tile_n` | **1536** | Must equal `base_moe_mlp_dim`. 1024 does not divide it and raises `AssertionError`; 512 divides it but cutting three ways is actually slower |
| `tile_k` | **2048** | The sweet spot. Not the 1024 from the table, and not "bigger is better" either (4096 OOMs outright) |
| `tile_m` | **512** | The table puts `m` in the 1024 bracket, but 512 measured 3.9% faster. **Copying the table is a good starting point, not the finish line** |

> The long-term fix is to run the official autotune and generate cache entries; injection is a verification device, but it captures the full gain.
> The complete 15-configuration sweep is in [TUNING-v7 §7.8.1](TUNING-v7.en.md#34-step-three-174-tokamax-tile--the-largest-single-item).

### 3.4 Running one round

```bash
XLA='--xla_tpu_scoped_vmem_limit_kib=65472 --xla_enable_async_all_gather=true --xla_tpu_enable_sparse_core_reduce_scatter_v2=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true --xla_tpu_enable_sparse_core_collective_aggregator=true --xla_tpu_use_tc_device_shape_on_sc=True --xla_sc_disable_megacore_partitioning=True --xla_tpu_enable_latency_hiding_layer_scheduler=true --xla_tpu_scheduler_percent_shared_memory_limit=150 --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true --xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false'

NP=64 bash hy3-run.sh "
export LIBTPU_INIT_ARGS='$XLA' JAX_PLATFORMS=tpu,cpu
export TK_TM=512 TK_TK=2048 TK_TN=1536
cd /deps && python3 -c 'exec(open(\"/tmp/tkcfg.py\").read());
  import runpy; runpy.run_module(\"src.maxtext.trainers.pre_train.train\", run_name=\"__main__\")' \
  src/maxtext/configs/base.yml model_name=hunyuan3-295b override_model_config=True \
  ici_data_parallelism=4 ici_fsdp_parallelism=128 ici_tensor_parallelism=1 \
  per_device_batch_size=12 max_target_length=4096 \
  megablox=True use_tokamax_gmm=True sparse_matmul=True use_custom_sort_vjp=True \
  scan_layers=True remat_policy=custom decoder_layer_input=offload out_proj=remat \
  attention=flash use_tokamax_splash=True sa_use_fused_bwd_kernel=True \
  sa_block_q=2048 sa_block_kv=2048 sa_block_kv_compute=2048 \
  sa_block_q_dkv=2048 sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=2048 \
  sa_block_q_dq=2048 sa_block_kv_dq=2048 \
  opt_type=adamw mu_dtype=bfloat16 grad_dtype=bfloat16 use_iota_embed=True \
  allow_split_physical_axes=True dtype=bfloat16 weight_dtype=float32 \
  tokenizer_type=tiktoken tokenizer_path=src/maxtext/assets/tokenizer_llama3.tiktoken \
  dataset_type=synthetic enable_checkpointing=False steps=8 jax_cache_dir=/tmp/jcn \
  base_output_directory=<GCS_OUT> run_name=myrun
"
```

**The 64-chip version** changes only two things: `NP=16`, and replace
`ici_data_parallelism=4 ici_fsdp_parallelism=128` with `ici_fsdp_parallelism=-1`.
**Keep `per_device_batch_size` at 12** — this value is optimal at both 64 and 256 chips (580 in each case).

**The 256-chip maximum version** (599) changes two more: `ici_data_parallelism=2 ici_fsdp_parallelism=256`, and
`per_device_batch_size=16`.

**You must clean up between rounds**, or the next one will not start:

```bash
NP=64 bash hy3-run.sh 'pkill -9 -f "pre_train[.]train"; rm -f /tmp/libtpu_lockfile'
```

> Write the pattern as `'pre_train[.]train'` rather than `'pre_train.train'` —
> the latter also matches the command line of the shell executing it, killing itself.

### 3.5 Collecting the numbers

```bash
grep -oE "completed step: [4-7], seconds: [0-9.]+"        $LOG   # step (steady state, take 4–7)
grep -oE "completed step: [4-7].*TFLOP/s/device: [0-9.]+"  $LOG   # per-device
grep -ohE "Total hbm usage >= [0-9.]+G"                    $LOG   # peak HBM
```

**Token throughput has to be computed yourself** (the framework does not report it directly):

```
tok/s        = device count × per_device_batch_size × max_target_length ÷ step
tok/s/chip   = tok/s ÷ (device count / 2)          # v7 has 2 devices per chip
```

Example: 512 devices × pdbs 16 × seq 4096 ÷ 30.40 s = **1,103,757 tok/s** = **4,312 tok/s/chip**.

**Unit conversion — the easiest thing to get wrong on v7. v7 has 2 devices per chip** (v5p is 1:1), and framework logs always report per device:

```
per-chip TFLOP/s = TFLOP/s/device from the log × 2
MFU              = per-chip ÷ 2307
```

> **Getting the direction backwards is a 4× error.** Always confirm the device : chip ratio before comparing MFU across generations.
> By the same token, the "device" in `per_device_batch_size` is **half a chip** on v7 —
> when comparing against v5p, the v7 side has to be halved to be equivalent.

**Two health checks you can run the moment it starts** (failing either means the code is not fully applied):

| Log field | Should be |
|---|---|
| `number parameters` | **298.786 billion** (digit-for-digit identical to v5p; constant across platforms) |
| `Total TFLOPs` | About **4547** at seq 4096. If it is roughly 5× that, the FLOP formula is missing `HUNYUAN3` and MFU will be inflated |

### 3.6 Reading logs — five rules that will save you

1. **Confirm all pods are Running before you read any log.** A TPU slice is all-or-nothing; when the set is incomplete, the surviving pods report
   `GetSliceInfo can only be invoked after a slice is built` — that is a symptom, not the cause.
2. **Judge errors from the earliest one, not the tail.** An invalid config brings the TPU up first and then exits, so the real error
   (`MAXTEXT CONFIG ERROR` / a pydantic `Value error`) is further up the log.
3. **Step 0 includes compilation, and steps 1/2 are artifacts of JAX's asynchronous dispatch**; take steady state from step ≥ 3.
4. **Logs must be persisted to disk.** Once a pod is deleted, preempted, or killed by the cluster's time limit, `kubectl logs` can never read them again.
5. 🔴 **On hitting `SLICE_FAILURE_*`, abort everything immediately and never retry.** This is a hardware fault;
   every restart crashes at the same layer, and once `maxRestarts` is exhausted the JobSet enters `Failed` —
   **with no consumer left, the autoscaler scales the nodes back to 0 and the chips are simply gone.**
   How to tell: `completed step: 0` has appeared at least once (so compilation and the execution path both work), peak HBM is far below the limit,
   and **a different worker crashes each time**. Hence set `maxRestarts` to 3, not 10.

```bash
# check after every round; exit on a hit
grep -q "SLICE_FAILURE" $LOG && { echo "TPU hardware fault, move to a different set of nodes"; exit 2; }
```

### 3.7 Diagnosing a managed (Kueue) cluster that won't scale up

**The real reason for a failed scale-up is not visible at the `kubectl` layer.** Pod events only ever give you
`Pod didn't trigger scale-up: ... in backoff after failed scale-up`. The real cause is inside the MIG:

```bash
gcloud container clusters describe <CLUSTER> --region <REGION> \
  --format='value(nodePools[].instanceGroupUrls)' | tr ';' '\n' | grep tpu
gcloud compute instance-groups managed list-errors <MIG> --zone <ZONE> \
  --format='value(error.code,error.message)'
```

One real output: `QUOTA_EXCEEDED  Quota 'HDB_TOTAL_GB' exceeded. Limit: 40960.0` —
**the regional disk quota was full**. TPU node boot disks are hyperdisk-balanced (100 GB each),
competing for the same regional quota as everyone else's PVCs; `Released` PVs left behind by others keep occupying it.

Diagnostic order (**looking at the pod directly will mislead you**):

| Step | What to look at | Getting stuck here means |
|---|---|---|
| ① | `kubectl get clusterqueue` / `get workload` | Quota accounting did not pass |
| ② | MIG `list-errors` | **Quota or capacity — machines cannot be created** |
| ③ | `kubectl get nodes` | Machines were created but did not register |
| ④ | `kubectl get pods` | Ordinary problems: image, config |

> **Being admitted ≠ having machines.** Passing the queue only means the accounting cleared.
> The converse holds too: **a full ledger ≠ busy machines** — we have seen accounting show 364/375 occupied while only 60 chips were actually running.

**A separate, independent trap: the `exclusive-topology` annotation makes the autoscaler see less demand than there is.**
It requires the leader pod to land first so that followers can copy its `gke-nodepool` selector.
With no free nodes the leader never lands, and the symptom is `parallelism=16` with `status.active=1` — **and no error message at all**.
Workarounds: hard-code `gke-nodepool` in the `nodeSelector`, or drop the annotation.

---

## 4. Measured data

### 4.1 Hardware

| | 64 chips | 256 chips |
|---|---|---|
| Nodes | 16 × `tpu7x-standard-4t` | 64 |
| Topology | `4x4x4` | `4x8x8` |
| JAX devices | **128** | **512** |
| HBM | 192 GB / chip; **94.74 GB usable per device** | same |
| BF16 peak / chip | **2,307 TFLOPS** (FP8 is 4,614) | same |
| Total BF16 compute | 147.6 PFLOPS | 590.6 PFLOPS |

### 4.2 From 457 to 599: what each step was worth

All measured on 256 chips in the same batch (same set of long-lived pods, compilation cache reused):

| Step | Increment | chip | MFU | tok/s | Peak HBM | Cumulative |
|---|---|---|---|---|---|---|
| Starting point | `DP4×FSDP128` / pdbs 8 / megablox | 453 | 19.64% | 835,131 | 74.20 G | — |
| **+tile** | tokamax `tile(512, 2048, 1536)` | **532** | 23.04% | 979,893 | 75.33 G | **+17.4%** |
| **+batch** | pdbs 8 → 10 | **564** | 24.45% | 1,039,573 | 84.06 G | **+24.5%** |
| **+batch** | pdbs 10 → 12 | **580** | 25.12% | 1,068,372 | 91.94 G | **+28.0%** |
| +change the base | `DP2×FSDP256` saves 13 G, pdbs → 14 | 585 | 25.37% | 1,078,802 | 89.56 G | +29.1% |
| **+more batch** | same base, pdbs → 16 | **599** | **25.96%** | **1,103,757** | 92.33 G | **+32.2%** |

The same recipe on 64 chips (`DP=1 × FSDP=128`, 16 nodes / 128 devices):

| Configuration | step | chip | MFU | tok/s | tok/s/chip | Peak HBM | Same config at 256 chips |
|---|---|---|---|---|---|---|---|
| megablox / pdbs 8 (old baseline, reproduced) | 19.92 s | 457 | 19.80% | 210,570 | 3,290 | 74.20 G | 453 |
| tile + pdbs 8 | 16.75 s | 543 | 23.55% | 250,343 | 3,912 | 75.33 G | 532 |
| tile + pdbs 10 | 20.28 s | 561 | 24.32% | 258,550 | 4,040 | **84.06 G** | 564 (HBM **84.06 G**) |
| **tile + pdbs 12** | **23.54 s** | **580** | **25.14%** | **267,284** | **4,176** | **91.94 G** | **580** (HBM **91.94 G**) |
| tile + pdbs 14 | — | model predicts 100.7 G → OOM, not measured | | | | | |

> **The two scales match point for point under the same recipe, and at pdbs 10 and 12 the peak HBM is identical to the byte
> (84.06 / 91.94 G).** This is the hardest evidence that "the DP dimension does not change the single-step computation graph."
>
> **The ceiling for 64 chips is exactly 580** — it only has 128 devices,
> so `FSDP=256` is not an option and it cannot free HBM by widening FSDP the way 256 chips can, which is what allows pdbs=14 there.

### 4.2.1 Scaling: weak scaling is 100%, strong scaling has a price

**The same 512 devices, two ways of scaling, an 11% gap.**

| Scaling mode | How the 512 devices are split | Work per device | global batch | per-chip | Relative to 64 chips |
|---|---|---|---|---|---|
| **Weak scaling** (add chips, add batch) | `DP=4 × FSDP=128` | unchanged (pdbs 12) | **4×** | **580** | **100.0%** |
| **Strong scaling** (add chips, keep batch) | `DP=1 × FSDP=512` | shrinks to 1/4 | 1× | 404 | 89% |

> The same recipe at 64 chips gives 580 (pdbs 12); at 256 chips it also gives 580.
> **Four times the chips, and per-chip throughput does not drop at all — weak scaling efficiency is 100%.**

#### Why the DP dimension can reach 100%

`DP=4 × FSDP=128` is just **four independent 64-chip jobs**. Within a group every layer performs FSDP
all-gather / reduce-scatter; **across groups there is exactly one gradient all-reduce in the entire step**.

Quantifying how cheap that all-reduce is:

```
Hy3 gradients (bf16)          ≈ 590 GB
per-device gradient shard      = 590 / 128 ≈ 4.6 GB      (FSDP=128)
ring all-reduce traffic        = 2(p−1)/p × 4.6 GB = 6.9 GB   (p = DP = 4)
v7 ICI per-chip bidirectional  = 1,200 GB/s
                               ──────────────────────────────
theoretical time               ≈ 12 ms   ← 0.05% of a 23.54 s step
```

**Even under a conservative 1/6 bandwidth-utilization assumption (35 ms), it is only 0.15%** — completely
buried inside the ±3% reproduction noise. And it can still overlap with the tail of the backward pass.

By contrast, **the intra-group FSDP collectives run twice per layer, which is 160 times across 80 layers** —
two orders of magnitude more. **This is the fundamental reason DP is cheap and FSDP is expensive.**

#### Why strong scaling loses 11%

`FSDP=512` spreads the same weights across 4× the chips, shrinking each shard to a quarter.
**The number of collectives is unchanged, but each one now carries only a quarter of the payload** —
the fixed overheads (synchronization, launch latency, multi-hop forwarding on the torus) no longer amortize.

> **In one sentence: when you add chips, add batch at the same time.**
> Adding chips without batch (spreading FSDP ever thinner) takes this model from 453 down to 404.

#### Three boundary conditions (do not over-extrapolate)

1. **We only measured up to DP=4.** DP=8 / 16 remain unverified. In theory ring all-reduce traffic
   `2(p−1)/p × N` approaches `2N` (a constant) as p grows, so **we expect it to stay close to 100%**,
   but that is an inference, not a measurement.
2. **This is a within-slice result.** All 512 devices live in one `4x8x8` slice and communicate over ICI.
   **DP across slices goes over DCN, whose bandwidth is more than an order of magnitude lower; the conclusion does not carry over.**
3. **It assumes the per-device workload stays constant.** If you hold the global batch fixed while scaling (true strong
   scaling), you degrade to the 404 row above.

---

### 4.2.2 How to compute HBM: a two-parameter model

You can predict the batch ceiling without hitting an OOM. Solve `HBM = static + slope × pdbs` from two measured points on the same base:

```
DP4×FSDP128:  74.20 G @ pdbs 8 , 91.93 G @ pdbs 12
              → static 38.7 G , slope 4.43 G / pdbs
DP2×FSDP256:  static 25.9 G (FSDP doubles, static halves), same slope
```

| Base | pdbs 8 | pdbs 10 | pdbs 12 | pdbs 14 | pdbs 16 |
|---|---|---|---|---|---|
| `DP4×FSDP128` predicted | 74.2 | 84.1 | 91.9 | 100.8 | 109.6 |
| `DP4×FSDP128` **measured** | **74.20** ✅ | **84.06** ✅ | **91.94** ✅ | — | **OOM** ✅ prediction correct |
| `DP2×FSDP256` predicted | 61.4 | 73.5 | 79.1 | 87.9 | 96.8 → predicted OOM |
| `DP2×FSDP256` **measured** | **61.36** ✅ | — | **78.27** ✅ | **89.56** ✅ | **92.33** ❌ **prediction wrong, it actually ran** |

> ⚠️ **This linear model breaks down beyond pdbs ≥ 14, and I got it wrong here a second time.**
>
> The measured piecewise slopes for `DP2×FSDP256`:
> `8→12` is **4.23 G/pdbs**, `12→14` is **5.65**, and `14→16` collapses to **1.39**.
> In the high-batch regime activation growth is markedly **sublinear** (most likely XLA changing its remat / offload scheduling under memory pressure),
> so linear extrapolation **systematically overestimates** and misjudges runnable configurations as OOM.
>
> **How to use it correctly**:
> 1. Interpolate only **near the measured range** (±2 pdbs); **do not extrapolate more than 4 pdbs out**
> 2. A configuration predicted to OOM **is still worth running once** — both of my OOM calls were wrong (pdbs 12 @ FSDP128, pdbs 16 @ FSDP256)
> 3. Only predictions that are "far past the limit" (e.g. 109.6 G, 15% over) can be ruled out directly — those were right both times

### 4.3 The complete parameter set

**Parallelism (the only part that varies with scale)**

```
# 64 chips
ici_fsdp_parallelism=-1
# 256 chips
ici_data_parallelism=4
ici_fsdp_parallelism=128
# common to both
ici_tensor_parallelism=1         # TP is useless; attention holds only 2% of the parameters
```

**MoE (including the largest single gain of this round)**

```
megablox=True
use_tokamax_gmm=True             # ← only meaningful when paired with the tile injection from tkcfg.py
sparse_matmul=True
use_custom_sort_vjp=True
# environment variables
TK_TM=512  TK_TK=2048  TK_TN=1536
```

**Batch and sequence**

```
per_device_batch_size=12         # use 10 on 64 chips; anything higher OOMs
max_target_length=4096           # measured seq 8192 + pdbs 4 to be equivalent (451 vs 453); no need to switch
```

**Attention**

```
attention=flash
use_tokamax_splash=True          # v7-specific
sa_use_fused_bwd_kernel=True     # ⚠️ this one must be False on v5p
sa_block_q=2048  sa_block_kv=2048  sa_block_kv_compute=2048
sa_block_q_dkv=2048  sa_block_kv_dkv=2048  sa_block_kv_dkv_compute=2048
sa_block_q_dq=2048  sa_block_kv_dq=2048
```

**Rematerialization / offload / optimizer**

```
scan_layers=True                 # also keeps compile time from growing with layer count
remat_policy=custom
decoder_layer_input=offload
out_proj=remat

opt_type=adamw
mu_dtype=bfloat16                # Adam first moment down to bf16
grad_dtype=bfloat16
use_iota_embed=True
```

Optimizer state drops from 16 B/param to 12 B/param. optax does not allow setting `nu_dtype` independently;
it always follows `weight_dtype`. **Master weights remain fp32.**

**Precision and baseline conditions**

```
dtype=bfloat16  weight_dtype=float32  allow_split_physical_axes=True
tokenizer_type=tiktoken
tokenizer_path=src/maxtext/assets/tokenizer_llama3.tiktoken

dataset_type=synthetic           # measuring throughput, not convergence
enable_checkpointing=False       # keeps I/O from perturbing the readings
steps=8                          # steady state from steps 4–7; 8 steps is enough when not capturing a profile
```

### 4.4 XLA flags (15 of them)

```
# basics (2)
--xla_tpu_scoped_vmem_limit_kib=65472
--xla_enable_async_all_gather=true

# SparseCore offload group (9) — worth ±0 on v7, kept only to stay aligned with the official recipe
--xla_tpu_enable_sparse_core_reduce_scatter_v2=true
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true
--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true
--xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true
--xla_tpu_enable_sparse_core_collective_aggregator=true      # ← this one cannot be removed
--xla_tpu_use_tc_device_shape_on_sc=True
--xla_sc_disable_megacore_partitioning=True

# scheduler group (4) — the only group that is worth anything, +6.6%
--xla_tpu_enable_latency_hiding_layer_scheduler=true
--xla_tpu_scheduler_percent_shared_memory_limit=150
--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true
--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false
```

> **Eight of those nine SparseCore flags really are worth zero, but the ninth
> (`collective_aggregator`) is a hard dependency of the layer scheduler** — removing it yields
> `INVALID_ARGUMENT: Latency hiding layer scheduler requires sparse core collective aggregator`.
> **Prune flags as groups, not one at a time.**

> ⚠️ **libtpu hard-fails on flags it does not recognize** (`Unknown command line flag`, and the process exits).
> **Re-validate the whole flag set whenever you change images.**

### 4.5 Smoke test (run this first after any code change)

```bash
NODES=1 TOPO=2x2x1 MODEL=hunyuan3-smoke STEPS=8 \
  bash run.sh smoke per_device_batch_size=1 max_target_length=2048
```

A 4-layer reduction whose structure is identical to the 295B (192 experts, top-8, sigmoid, expert bias, shared expert,
GQA, QK-norm, fp32 routing, MTP — all at full spec); only the layer count is cut.

| | v7 measured | v5p, same command |
|---|---|---|
| Parameter count | **16.139 B** | 16.139 B (**must match**) |
| `total_weights` | 16384 | 8192 |
| loss (8 steps) | 13.411 → 11.091 | 13.453 → 10.354 |
| NaN / skipped | 0 | 0 |

> The two platforms producing different loss sequences **is correct**: with the same 4 chips, v7 has 8 devices while v5p has only 4,
> so at `pdbs=1` the global batch differs by 2×. **The only cross-platform constant that can serve as a hard check is the parameter count, 16.139 B.**

**Why 4 layers can stand in for 80**: MaxText groups by **type** when applying `scan`, so all 79 MoE layers share a single
compiled artifact and differ only in weight values. The smoke test exercises **that one function which is reused 79 times**.
What it cannot cover: memory pressure, large-scale sharding, the complete XLA flag set, convergence quality, and all performance.

---

## 5. Verification log

> **Every number in this document was produced by following its own steps.** Below is the record of each audit round's deviations —
> including the errors that got fixed, because "where the document will trip you up" is worth more than "what the document says."

### Round 1 (2026-08-04 23:16–23:30): both scales re-measured at once

**Method**: pods were not destroyed (see the note below); `hy3-run.sh` and `tkcfg.py` were **extracted verbatim** from this document's
markdown by script, and the training command was assembled per §3.4. **No historical script was used.**

| Scale | Recipe | Document says | Measured | Deviation |
|---|---|---|---|---|
| 64 chips | `DP1×FSDP128` + tile + pdbs 12 | step 23.54 / **580** | step **23.538** / **579.97** | 0.005% |
| 256 chips | `DP2×FSDP256` + tile + pdbs 16 | step 30.40 / **599** | step **30.399** / **598.74** | 0.04% |

Smoke test: 64 chips → 128 devices ✅ / 256 chips → 512 devices ✅

**Four defects caught:**

| # | Defect | Resolution |
|---|---|---|
| 1 | §0 said `pdbs=12` for 64 chips while §3.4 said "change it to 10" — **an internal contradiction**, left over from an older version | ✅ fixed |
| 2 | Deleting the JobSet to rebuild from scratch got **the whole pool taken by another job within 30 seconds**; the audit was interrupted and the chips could not be recovered | ✅ turned into a rule: audits do not re-run the capacity-acquisition step |
| 3 | `hy3-run.sh` was `for...& done; wait; cat`, so **the log was 0 bytes until the run finished**, making it impossible to tell compilation from a hang for 6–30 minutes | ✅ changed to live `tee` on worker-0 |
| 4 | The §3.2 smoke test only gave the expected value for 64 nodes (512), not the 128 for 16 nodes | ✅ added a table covering both scales |

> **On "not destroying the pods"**: performing a "wipe and rebuild from zero" audit on a shared cluster risks losing the chips outright
> (defect #2 is exactly that happening). And the "acquire resources" step **has nothing auditable in it** —
> its success depends on whether the cluster had free capacity at that moment, not on whether the document is correct.
> **What actually needs auditing is the smoke test, the injection, the training command, the number collection, and the cleanup** — all of which live inside the pods and need no environment rebuild.

---

## 6. Known limitations

| Item | Status |
|---|---|
| **0.2% short of target** | 599 vs 600–630. Remaining candidates in [TUNING-v7 §7.8.3, still to do](TUNING-v7.en.md#6-not-yet-tried) |
| Dataset | `synthetic`. **A falling loss only proves "it computes and does not diverge"; it is not evidence of convergence** |
| **FP8 + QAG (converged)** | With QAG on at 64 chips: **625** (vs 594 without QAG, **+5.3%**, and at a smaller batch); 256 chips without QAG: 618. ⚠️ "The tile on that kernel has never been swept, potential 726" **has been disproven** — FP8 still routes into tokamax internally and the tile has been in effect all along. The eight-cell experiment on 2026-08-05 produced not one positive result; **the tuning space is exhausted**. See [TUNING-v7 §4.6](TUNING-v7.en.md#46-what-can-be-tuned-and-what-cannot--one-master-table) |
| `shard_exp_on_fsdp` | `IndivisibleError` on 128 devices (192 % 128 ≠ 0); unusable |
| HF weights → Orbax conversion | Not done. Not needed for throughput alone; mandatory for SFT |
| Full loss curve | Not recorded. Worth adding one over 30+ steps |
| Capacity | tpu7x is in demand and the machine type exists in only 4 zones worldwide; see §2.6 |

---

## Appendix A: Quick reference of differences from v5p

The two platforms **share the same code and the same image**; the following is the complete set of differences:

| | v7 | v5p |
|---|---|---|
| Machine type / topology | `tpu7x-standard-4t`, `4x4x4` / `4x8x8`, us-central1-**c** | `ct5p-hightpu-4t`, `4x8x8`, us-central1-**a** |
| Node-pool prerequisite | **A workload policy must be created first** | none |
| Disk | **hyperdisk-balanced** | default |
| **device : chip** | **2 : 1** | 1 : 1 |
| MFU denominator | **2,307** | 459 |
| `max_target_length` | **4096** | 8192 |
| `sa_use_fused_bwd_kernel` | **True** | **False** |
| `use_tokamax_splash` | **True** | not set |
| `use_tokamax_gmm` + tile injection | **True, +17.4%** | negative gain, not used |
| `opt_type` / `mu_dtype` / `grad_dtype` | **adamw / bf16 / bf16** | default / fp32 / fp32 |
| MoE tile parameters (18 of them) | not set (tokamax tile injection used instead) | all set |
| XLA flags | 15 | 25 |
| SparseCore offload group gain | **±0** | **+4.07 pp** |
| **MFU** | **25.96% (256 chips) / 25.14% (64 chips)** | 35.07% (converged) |

> **The same switch can flip sign between the two platforms.** `sa_use_fused_bwd_kernel`,
> the SparseCore offload group, and `use_tokamax_gmm` are all examples. **Do not carry a tuning conclusion from one platform straight over to the other.**

---

## Appendix B: Further reading

| Document | Contents |
|---|---|
| [TUNING-v7.md](TUNING-v7.en.md) | Tuning in practice: why this water line, every ablation table, the failures, the HBM model |
| [QUICKSTART-v5p.md](QUICKSTART-v5p.md) 🇨🇳 | The v5p version, including the full architectural breakdown; baseline 35.07%, verified from scratch |
| [EXPERIMENT-LOG.md](EXPERIMENT-LOG.md) 🇨🇳 | The complete experiment archive: every round, post-mortems on 12 bugs, DWS flex-start node-pool creation |
| [MAXTEXT-PORTING-GUIDE.md](MAXTEXT-PORTING-GUIDE.md) 🇨🇳 | The general pattern for porting another model into MaxText |
| [maxtext-hunyuan3/](maxtext-hunyuan3/) | `prep.sh` / `run.sh` |
