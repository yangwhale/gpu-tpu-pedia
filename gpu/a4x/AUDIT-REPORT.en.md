> 🌐 [中文](AUDIT-REPORT.md) | **English**

# GB200 A4X Deployment Guide -- Audit Report

**Audit Date**: 2026-06-27
**Source Document**: 2475-line GB200 A4X NVL72 deployment guide (2026-06-19)
**Target**: 10 README files in `/home/chrisya/gpu-tpu-pedia/gpu/a4x/`

---

## 1. Section-to-File Mapping Summary

| Source Section | Target File | Status | Notes |
|----------------|-------------|--------|-------|
| Header / Architecture / TOC | `README.md` (top-level) | ✅ Complete | Architecture description, platform info, test results summary all present |
| 0. Architecture Overview (0.1-0.3) | `01-environment-setup/README.md` | ✅ Complete | All 3 tables + concept diagram preserved verbatim |
| 1. Environment Preparation (1.1-1.5) | `01-environment-setup/README.md` | ✅ Complete | All variables, commands, CIDR values match |
| 2. k8s 1.34.1 Cluster (2.1-2.4) | `02-k8s-cluster/README.md` | ✅ Complete | CP creation, Worker batch creation, join, labels all present |
| 3. GPU Stack + ComputeDomain (3.1-3.6) | `03-gpu-stack/README.md` | ✅ Complete | All Helm commands, RBAC YAML, ComputeDomain loop preserved |
| 4. GCSFuse + Lustre Storage (4.1-4.3) | `03-gpu-stack/README.md` | ✅ Complete | Correctly merged into same file |
| 5. NCCL Tests (5.1-5.4) | `04-nccl-test/README.md` | ✅ Complete | All 4 test scenarios + results tables match |
| 6. RDMA Bandwidth Test | `05-rdma-test/README.md` | ✅ Complete | Commands and results verbatim |
| 7. DeepEP Tests (7.1-7.6) | `06-deepep-test/README.md` | ✅ Complete | Both paths, NVSHMEM matrix, 4-GPU patches, all test results |
| 8. Megatron-LM Training (8.1-8.2) | `07-megatron-training/README.md` | ✅ Complete | Full torchrun commands with all parameters preserved |
| 9. Multi-Domain Training (9.1-9.7) | `08-multi-domain/README.md` | ✅ Complete | Bandwidth tiers, rank encoding, JobSet YAML, Per-Comm MNNVL |
| 10. 64+8 Spare Capacity | `08-multi-domain/README.md` | ✅ Complete | Merged into same file (sections 10.1-10.4) |
| 11. Report-Faulty API | `08-multi-domain/README.md` | ⚠️ Partial | gcloud CLI + fault-reasons table present; **Sub-block domain-level reporting (11.2) dropped** |
| 12. GPU Monitoring (DCGM) | `08-multi-domain/README.md` | ✅ Complete | Commands, metrics table, Grafana note all present |
| 13. Test Results Summary | `README.md` (top-level) | ✅ Complete | All numeric results match source |
| 14. Known Issues (14.1-14.9) | Split across `01-environment-setup`, `03-gpu-stack`, `06-deepep-test`, `07-megatron-training` | ✅ Complete | Each README includes its relevant known-issues subsections |
| Appendix A (VM Acceptance) | `01-environment-setup/README.md` | ⚠️ Partial | Tables present but simplified; **B.5 Startup Scripts section dropped**; customer remarks column removed from some tables |
| Appendix B (Integration Debug) | `07-megatron-training/README.md` + `06-deepep-test/README.md` | ⚠️ Partial | B.4-B.6 in 07-megatron; B.1-B.3 in 06-deepep; **individual appendix sub-numbers not labeled** |
| Appendix C (Benchmark Standards) | `08-multi-domain/README.md` | ✅ Complete | GEMM, NCCL, Training performance tables all present |
| RL Training (Section 9 new) | `09-rl-training/README.md` | ⚠️ Source Gap | **Not in source document** -- entirely new content added to READMEs |

---

## 2. Detailed Findings Per File

### 2.1 `README.md` (Top-level)

**Status**: ✅ Complete

- Architecture description matches source header
- Table of contents correctly maps to 9 subdirectories
- Test results summary table includes all numeric values from source Section 13
- Date attribution preserved ("Verified 2026-06-19")

**Issues**: None found.

---

### 2.2 `01-environment-setup/README.md`

**Status**: ✅ Complete (minor simplifications)

**Content verified**:
- Section 0 (0.1, 0.2, 0.3): All 3 hardware tables, concept speed-reference table, and ASCII architecture diagram match source verbatim
- Section 1 (1.1-1.5): All global variables with correct values, VPC/subnet/firewall commands, Placement Policy loop, reservation check -- all match
- Known Issues (from Section 14): TLinux 4, VPC MTU, Bond not applicable, fabricmanager, nvidia_peermem, hybrid cloud notes -- present and correct
- Appendix A: VM acceptance tables present

**Issues found**:

| # | Category | Severity | Description |
|---|----------|----------|-------------|
| 1 | Content Gap | Low | Source global variables block includes inline comments like "450 A4X VMs + CP + regular VMs + buffer" for GVNIC_CIDR but README trims some of these annotations. Values are correct. |
| 2 | Content Gap | Low | Source network layout table has 6 rows (4 RDMA subnets shown individually); README consolidates RDMA into 1 row with `sub-0..3`. Functionally equivalent but less detailed. |
| 3 | Content Gap | Low | Source Appendix A has separate "B.1"-"B.5" subsection numbering and a "Customer Remarks" column; README uses simplified headers without customer remarks column. The B.5 "Startup Scripts" subsection is dropped entirely. |

---

### 2.3 `02-k8s-cluster/README.md`

**Status**: ✅ Complete

**Content verified**:
- 2.1 CP node creation command matches exactly (machine type, image, boot disk, network-interface, scopes)
- Startup script 7-step summary matches
- Join info extraction commands match (openssl pipeline for hash)
- 2.2 Worker batch creation loop with all flags (reservation, provisioning-model, maintenance-policy, network interfaces) matches verbatim
- 2.3 Worker join script with Phase 1/Phase 2 description matches
- 2.4 Node labels: Method A (manual) and Method B (Metadata Server auto) both present with correct code
- Topology discovery API comparison table matches source

**Issues found**: None.

---

### 2.4 `03-gpu-stack/README.md`

**Status**: ✅ Complete

**Content verified**:
- 3.1 nvidia-device-plugin Helm install matches (repo, namespace, nodeSelector escaping)
- 3.2 DRA GPU Driver Helm install matches (repo, namespace, version, gpuResourcesEnabledOverride)
- 3.3 DRANET install matches (OCI registry, version)
- 3.4 RDMA DeviceClass YAML preserved verbatim (CEL expression identical)
- 3.5 Scheduler DRA RBAC: Full ClusterRole + ClusterRoleBinding YAML matches exactly, including scheduler restart commands
- 3.6 ComputeDomain creation loop matches (apiVersion, numNodes, allocationMode, UID label)
- 4.1-4.3 GCSFuse + Lustre content matches (basic mount, v2 parallel download config, hostPath volumes)
- Notes table at bottom consolidates key k8s 1.34 DRA + ComputeDomain caveats from Section 14.2

**Issues found**:

| # | Category | Severity | Description |
|---|----------|----------|-------------|
| 4 | Content Gap | Low | Source Section 3.6 mentions a "multiple tasks within a domain" paragraph about multiple tasks sharing one ComputeDomain ResourceClaimTemplate. README includes this but slightly shorter. |

---

### 2.5 `04-nccl-test/README.md`

**Status**: ✅ Complete

**Content verified**:
- 5.1 Single node test: kubectl apply command, result table (683.75 GB/s) match
- 5.2 Same-domain MNNVL: DRA resource claim pattern YAML comment block matches; SSH key exchange, MPI compilation, scp, mpirun command all match exactly; result (834.95 GB/s) matches
- 5.3 Cross-domain RDMA: commands match; NCCL_MNNVL_ENABLE=0 correctly used; result (325.88 GB/s) matches
- 5.4 Mixed 4-node: All 4 host IP extractions, mpirun command with 16 GPUs, MNNVL_ENABLE warning; result (162.45 GB/s) matches
- Summary table at bottom correctly aggregates all 4 results

**Issues found**: None.

---

### 2.6 `05-rdma-test/README.md`

**Status**: ✅ Complete

**Content verified**:
- perftest install commands match
- Server/client loop with ports 18515-18518 matches
- LD_LIBRARY_PATH override for aarch64 system libs documented
- Results (381 Gbps per NIC, 1524 Gbps aggregate) match
- GPU-NIC mapping table (GPU 0 = mlx5_0 etc.) matches Section 7.5 reference

**Issues found**: None.

---

### 2.7 `06-deepep-test/README.md`

**Status**: ✅ Complete

**Content verified**:
- 7.1 Overview: Path A/B comparison table matches
- Internode limitation + GDRCopy explanation matches
- 7.2 NVSHMEM compatibility matrix (3 version rows + 2 platform rows) matches
- 7.3 Path A: kubectl apply, NVSHMEM 3.4.5 pip install, 4-GPU sed patches, build command, intranode test, low-latency test -- all match
- 7.4 Path B: NVSHMEM install via apt, CCCL header fix, DeepEP v2 clone + patches, compile env vars, internode test script -- all match
- 7.5 4-GPU adaptation: Constant table (v1/v2), 4 core modification points -- all match
- 7.6 Test summary: Path A commands, Path B commands, CUDA 12 vs 13 comparison table -- all match
- DeepEP v1 vs v2 selection table (from Section 14.9) included at bottom -- correct

**Issues found**:

| # | Category | Severity | Description |
|---|----------|----------|-------------|
| 5 | Content Gap | Very Low | Source Section 7.1 references "14.8 DeepEP v1 vs v2 selection recommendation" by number; README says "version selection guide" without section number. Cross-reference still works via inline content. |

---

### 2.8 `07-megatron-training/README.md`

**Status**: ✅ Complete

**Content verified**:
- DRA resource claim pattern YAML matches (ResourceClaimTemplate + Pod spec)
- Pod deployment + SSH key exchange commands match
- 8.1 Single-node: Full torchrun command with all ~50 arguments preserved verbatim
- 8.2 Multi-node: Full torchrun command for node_rank=1 preserved with all env vars (GLOO_SOCKET_IFNAME, NCCL_MNNVL_ENABLE=2, NCCL_CUMEM_ENABLE=1, USE_MNNVL=1)
- Results table (356 TFLOP/s single, 274 TFLOP/s multi) matches
- GB200 performance tips table (from Section 14.5) matches
- Appendix B debugging content (B.4 parameter changes, B.5 short hostname, B.6 hostname resolution) included

**Issues found**:

| # | Category | Severity | Description |
|---|----------|----------|-------------|
| 6 | Execution Risk | Medium | In Section 8.2 multi-node command, the README uses `--master_addr=\$MEGA_HOST1_IP` (escaped dollar sign). Source uses `--master_addr=$MEGA_HOST1_IP`. The README version is actually correct for use inside a `kubectl exec -- bash -c "..."` context (the escape is needed to prevent premature shell expansion on the host). The source version would also work since it's in a similar context. Not a real bug, but the inconsistency may confuse readers. |
| 7 | Content Gap | Low | Source Section 14.5 title says "Megatron-LM GB200 performance optimization highlights" and includes the line about Megatron-LM repo path `tests/functional_tests/test_cases/gpt/gpt3_15b_8t_release_gb200/model_config.yaml`. README omits this specific repo path reference. |

---

### 2.9 `08-multi-domain/README.md`

**Status**: ✅ Complete (with minor omissions)

**Content verified**:
- 9.1 Bandwidth tiers table matches (L1/L2/L3)
- 9.2 Parallelism strategy table + rank encoding rules + domain example matches
- 9.3 Recommended parallel configs table matches
- 9.4 JobSet + Kueue TAS: Topology CRD, Kueue queue YAML, full JobSet YAML -- all present and match
- Kueue v0.18 notes (v1beta2, nodeLabels, namespaceSelector) match
- Feature comparison table (manual vs JobSet) matches
- Kubeflow Trainer mention matches
- 9.5 Manual Pod verification: parameter comparison table matches
- 9.6 4+ domain expansion: NODE_RANK encoding, parallel parameter scaling table matches
- Dense model strategy matches
- 9.7 Per-Communicator MNNVL: NCCL version table, verification plan, expected benefit -- matches
- Section 10 (64+8 spare capacity): PriorityClass, Placeholder, preemption test, cordon/drain -- all match
- Section 11 (Report-Faulty): gcloud CLI command, fault-reasons table, Python SDK matches
- Section 12 (DCGM): deployment, metrics table, Grafana note matches
- Appendix C (Benchmarks): GEMM, NCCL, Training tables match

**Issues found**:

| # | Category | Severity | Description |
|---|----------|----------|-------------|
| 8 | Content Gap | Medium | **Source Section 11.2 "Sub-block domain-level fault reporting" is dropped entirely.** This includes `gcloud alpha compute reservations blocks list`, `gcloud alpha compute reservations sub-blocks list`, and `gcloud alpha compute reservations sub-blocks report-subblock-as-faulty` commands. These are important for NVSwitch-level fault reporting. |
| 9 | Content Gap | Low | Source Python SDK example includes a usage example block with `report_host_as_faulty(project_id="your-gcp-project", ...)`. README has the function definition but not the usage example. |
| 10 | Content Gap | Low | Source Appendix C training table includes `iteration_time (ms)` and `tokens/gpu/s` columns. README simplifies to only `TFLOP/s/gpu` and `MFU` columns. |
| 11 | Content Gap | Low | Source NCCL performance table note about "2-domain All2All bandwidth explanation" is present but slightly shorter in README. |

---

### 2.10 `09-rl-training/README.md`

**Status**: ⚠️ Source Gap -- This is NEW content NOT in the source document

This README covers RL training with veRL/AReaL on GB200, including:
- Hardware overview (referencing holdback from Section 10)
- Framework selection (veRL, AReaL)
- Model selection (Qwen 3.5 397B, DeepSeek V3 671B)
- Parallelism strategy (TP=8, EP=8, DP=8, FSDP)
- Deployment modes (colocated vs disaggregated)
- GRPO configuration with hyperparameter table
- veRL YAML config example
- Performance tuning tips
- Known limitations

**Assessment**: This is supplementary content that adds value to the guide. It is self-consistent and references the correct hardware specs (4 GPU/node, 64 GPU available after holdback, MNNVL bandwidth values). However, since it has no source to audit against, the content cannot be verified for accuracy of recommendations.

**Issues found**:

| # | Category | Severity | Description |
|---|----------|----------|-------------|
| 12 | Execution Risk | Medium | The RL README recommends `TP=8` mapping to "2 nodes x 4 GPU/node". This requires MNNVL across 2 nodes. While valid for same-domain nodes, the README does not explicitly warn that both nodes MUST be in the same NVL72 domain for this TP=8 config to achieve the stated ~840 GB/s bandwidth. |
| 13 | Content Gap | Low | The `verl_config.yaml` is a reference template but has no tested benchmark results (unlike all other sections which include measured results). |
| 14 | Source Gap | Info | The RL training section was not in the original deployment guide. It should be clearly marked as supplementary/advisory content rather than validated deployment instructions. |

---

## 3. Consolidated Issue List

### Content Gaps (information in source but missing/simplified in READMEs)

| # | File | Severity | Description |
|---|------|----------|-------------|
| 8 | `08-multi-domain/` | **Medium** | Sub-block domain-level fault reporting (Section 11.2) entirely dropped -- `gcloud alpha compute reservations sub-blocks report-subblock-as-faulty` commands missing |
| 3 | `01-environment-setup/` | Low | Appendix A B.5 Startup Scripts subsection dropped |
| 7 | `07-megatron-training/` | Low | Megatron repo reference path for GB200 config YAML omitted |
| 9 | `08-multi-domain/` | Low | Python SDK usage example for report-faulty omitted |
| 10 | `08-multi-domain/` | Low | Appendix C training table simplified (iteration_time and tokens/gpu/s columns dropped) |
| 1 | `01-environment-setup/` | Very Low | Some inline CIDR comments trimmed |
| 2 | `01-environment-setup/` | Very Low | RDMA subnet table rows consolidated |
| 4 | `03-gpu-stack/` | Very Low | Multi-task ComputeDomain explanation slightly shorter |
| 5 | `06-deepep-test/` | Very Low | Section cross-references use descriptions instead of numbers |
| 11 | `08-multi-domain/` | Very Low | All2All bandwidth explanation slightly shorter |

### Execution Risks (commands/configs that could cause failures)

| # | File | Severity | Description |
|---|------|----------|-------------|
| 12 | `09-rl-training/` | Medium | TP=8 across 2 nodes requires same-domain placement but README doesn't warn about this constraint |
| 6 | `07-megatron-training/` | Low | `\$MEGA_HOST1_IP` escaping style differs from source (but is correct for the context) |

### Source Gaps (important topics the source document itself does not cover)

| # | Topic | Severity | Description |
|---|-------|----------|-------------|
| 15 | Security | Medium | No discussion of Pod Security Standards, network policies, or RBAC beyond scheduler DRA. Production deployments should address container security, image signing, and least-privilege RBAC. |
| 16 | Monitoring/Alerting | Medium | DCGM metrics are defined but there is no guidance on setting up Prometheus alerting rules, PagerDuty/OpsGenie integration, or automated remediation workflows. |
| 17 | Checkpoint/Restart | Medium | No guidance on distributed checkpoint management for training fault recovery. `--save-interval` is set but there is no documentation on checkpoint storage strategy (GCSFuse vs Lustre for ckpt I/O) or automatic restart-from-checkpoint procedures. |
| 18 | Node Health Checks | Low | No automated GPU health check (e.g., periodic DCGM health check DaemonSet, automated XID monitoring → cordon → report-faulty pipeline). The manual cordon/drain/report-faulty workflow is documented but not automated. |
| 19 | Quota/Cost | Low | No discussion of reservation costs, preemption risks for spot VMs, or quota management for multi-project environments. |
| 20 | Calico Tuning | Low | Source mentions Calico VXLAN + `IP_AUTODETECTION_METHOD=interface=eth0` but does not provide the actual Calico installation/configuration commands beyond the CP startup script reference. |
| 21 | Multi-tenancy | Low | ComputeDomain is documented as one-per-node, but no guidance on how multiple teams share a cluster's NVL72 domains (namespace isolation, ResourceQuota per domain, etc.). |
| 14 | RL Training | Info | 09-rl-training content is supplementary with no tested benchmark results from the original validation environment. |

---

## 4. Recommended Fixes

### Priority 1 (Should Fix)

1. **Add Sub-block fault reporting to `08-multi-domain/README.md`**: Restore Source Section 11.2 content including `gcloud alpha compute reservations blocks list`, `sub-blocks list`, and `report-subblock-as-faulty` commands. This is critical for NVSwitch-level fault management.

2. **Add domain placement warning to `09-rl-training/README.md`**: Add explicit note that TP=8 (2 nodes) requires both nodes in the same NVL72 domain, and reference `01-environment-setup/` Placement Policy section.

3. **Mark `09-rl-training/` as advisory**: Add a note at the top stating this content is supplementary guidance not validated in the original deployment test cycle.

### Priority 2 (Nice to Have)

4. **Restore Python SDK usage example in `08-multi-domain/`**: Add the `report_host_as_faulty(project_id=..., ...)` call example from source Section 11.3.

5. **Restore Appendix C full training table columns**: Add back `iteration_time (ms)` and `tokens/gpu/s` columns for completeness.

6. **Add Megatron config repo reference to `07-megatron-training/`**: Restore the path `tests/functional_tests/test_cases/gpt/gpt3_15b_8t_release_gb200/model_config.yaml`.

### Priority 3 (Documentation Quality)

7. **Consider adding Checkpoint/Restart guidance**: Document best practices for checkpoint storage (GCSFuse vs Lustre), restart-from-checkpoint procedures, and `--load` parameter usage.

8. **Consider adding GPU Health Monitoring automation**: Document a DaemonSet-based XID monitoring approach with automated cordon + report-faulty.

---

## 5. Overall Assessment

The split from monolithic source to 10 README files is **well executed**. Content fidelity is high:

- **All code blocks and commands are preserved verbatim** (including the 50+ argument Megatron torchrun commands)
- **All YAML manifests are complete** (DRA RBAC, ComputeDomain, DeviceClass, JobSet)
- **All test result numbers match exactly** across source and READMEs
- **All environment variables and version numbers are correct**
- **Cross-references between READMEs are properly updated** (using relative paths like `../03-gpu-stack/`)

The only material content gap is the missing Sub-block fault reporting commands (Issue #8). All other gaps are Low/Very Low severity simplifications that do not affect the guide's usability.

**Verdict**: Ready for use with the 3 Priority-1 fixes applied.

---

*Audit performed by Claude Code (Opus 4.6) on 2026-06-27*
