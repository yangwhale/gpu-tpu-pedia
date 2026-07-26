#!/bin/bash
# 把 Megatron-Bridge main 分支的 HYV3Bridge 单文件移植进 r0.5.0 容器。
#
# 背景：Hy3 的 bridge 只存在于 main 分支，v0.5.0 / v0.5.1 两个 release 都没有
#      （`gh api .../contents/.../hy_v3?ref=v0.5.1` 返回 404）。
#      但它只依赖 r0.5.0 就有的稳定接口，所以不必升级整个 Bridge。
#
# 用法：./install_hy3_bridge.sh <pod名> [pod名...]
# 校验：脚本会打印容器内外的 md5，两边必须一致（base64 over kubectl exec 曾丢过内容）。
set -euo pipefail
CTX=${CTX:-gke_tencent-gcp-taiji-poc_us-central1_gb300-gke-test}
REF=${REF:-main}
PODS=${*:?用法: $0 <pod名> [pod名...]}

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
for f in hy_v3_bridge.py __init__.py; do
  gh api "repos/NVIDIA-NeMo/Megatron-Bridge/contents/src/megatron/bridge/models/hy_v3/$f?ref=$REF" \
     --jq '.content' | base64 -d > "$TMP/$f"
done
LOCAL_MD5=$(md5sum "$TMP/hy_v3_bridge.py" | cut -d' ' -f1)
echo "本地 md5: $LOCAL_MD5"

B1=$(base64 -w0 "$TMP/hy_v3_bridge.py")
B2=$(base64 -w0 "$TMP/__init__.py")
cat > "$TMP/inst.sh" <<INSTEOF
#!/bin/bash
set -e
D=/opt/Megatron-Bridge/src/megatron/bridge/models
mkdir -p \$D/hy_v3
echo $B1 | base64 -d > \$D/hy_v3/hy_v3_bridge.py
echo $B2 | base64 -d > \$D/hy_v3/__init__.py
python3 - <<'PY'
p = "/opt/Megatron-Bridge/src/megatron/bridge/models/__init__.py"
s = open(p).read()
if "hy_v3" in s:
    print("already registered"); raise SystemExit(0)
anchor = "from megatron.bridge.models.gpt_provider import"
assert anchor in s, "import anchor not found"
s = s.replace(anchor, "from megatron.bridge.models.hy_v3 import HYV3Bridge  # noqa: F401\n" + anchor, 1)
a2 = '__all__ = ['
assert a2 in s, "__all__ not found"
s = s.replace(a2, a2 + '\n    "HYV3Bridge",', 1)
open(p, "w").write(s)
print("patched models/__init__.py")
PY
md5sum \$D/hy_v3/hy_v3_bridge.py
INSTEOF

B=$(base64 -w0 "$TMP/inst.sh")
for p in $PODS; do
  echo "=== $p ==="
  kubectl --context "$CTX" exec "$p" -- bash -c \
    "echo $B | base64 -d > /tmp/inst.sh && bash /tmp/inst.sh" 2>&1 | grep -v '^Defaulted'
done
echo "对照本地 md5: $LOCAL_MD5"
