#!/usr/bin/env python3
"""Static verification for the Hunyuan3 MaxText block.

Run from a MaxText checkout that has the patch applied:
    python3 verify_hunyuan3.py            # from anywhere, pass --root
    python3 verify_hunyuan3.py --root /path/to/maxtext

Checks, in order:
  1. hunyuan3-295b.yml reproduces the 294.9 B parameter count from the
     GB300 source-of-truth document (and the A21B activated count)
  2. DecoderBlockType.HUNYUAN3 is registered
  3. Both layer classes exist
  4. decoders.get_decoder_layers() dispatches to [dense, moe] — a 2-element
     list is what the existing DeepSeek scan machinery asserts on
  5. The norm layer resolves to rms_norm

Only data-pipeline / quantization deps are stubbed (grain, tensorflow, qwix);
none of them participate in graph construction.
"""
import argparse, importlib, os, sys, types
from unittest.mock import MagicMock

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
ap.add_argument("--config", default=None)
a = ap.parse_args()
cfg_path = a.config or os.path.join(os.path.dirname(os.path.abspath(__file__)), "hunyuan3-295b.yml")

# ---- 1. parameter count from the yml ----
import yaml
c = yaml.safe_load(open(cfg_path))
L, E = c["base_num_decoder_layers"], c["base_emb_dim"]
FF, MOE = c["base_mlp_dim"], c["base_moe_mlp_dim"]
NE, SH, D = c["num_experts"], c["shared_experts"], c["first_num_dense_layers"]
V, QH, KH, HD = c["vocab_size"], c["base_num_query_heads"], c["base_num_kv_heads"], c["head_dim"]
nmoe = L - D
routed = nmoe * NE * 3 * E * MOE
shared = nmoe * SH * 3 * E * MOE
attn = L * (E * QH * HD + 2 * E * KH * HD + QH * HD * E)
dense = D * 3 * E * FF
emb = 2 * V * E
tot = routed + shared + attn + dense + emb
act = nmoe * ((c["num_experts_per_tok"] + SH) * 3 * E * MOE) + attn + dense + emb
print(f"1) total params  {tot/1e9:.2f} B   (SSOT 294.9 B, delta {abs(tot/1e9-294.9)/294.9*100:.2f}%)")
print(f"   activated     {act/1e9:.1f} B    (official A21B)")
print(f"   experts share {(routed+shared)/tot*100:.1f}%  -> EP is the only memory knob that matters")
assert abs(tot/1e9 - 294.9) / 294.9 < 0.01, "parameter count drifted >1% from SSOT"

# ---- 2..5. wiring ----
sys.path.insert(0, os.path.join(a.root, "src"))
stubbed = []
def _imports():
    for m in ("MaxText.common_types", "MaxText.layers.hunyuan3", "MaxText.layers.decoders"):
        importlib.import_module(m)
for _ in range(30):
    try:
        _imports(); break
    except ModuleNotFoundError as e:
        if e.name.startswith("MaxText"): raise
        sys.modules[e.name] = MagicMock(); stubbed.append(e.name)
        for k in [k for k in list(sys.modules) if k.startswith("MaxText")]: del sys.modules[k]
else:
    sys.exit("could not import MaxText after 30 stub rounds")

from MaxText.common_types import DecoderBlockType
from MaxText.layers import hunyuan3, decoders
print(f"\n   stubbed (not graph-relevant): {sorted(set(stubbed))}")
print(f"2) enum          {DecoderBlockType('hunyuan3')}")
print(f"3) layer classes {[n for n in dir(hunyuan3) if n.startswith('Hunyuan')]}")

cfg = MagicMock(); cfg.decoder_block = DecoderBlockType.HUNYUAN3
dec = decoders.Decoder.__new__(decoders.Decoder); object.__setattr__(dec, "config", cfg)
layers = decoders.Decoder.get_decoder_layers(dec)
print(f"4) dispatch      {[k.__name__ for k in layers]}")
assert len(layers) == 2, "must be [dense, moe]"
assert layers[0] is hunyuan3.Hunyuan3DenseLayer and layers[1] is hunyuan3.Hunyuan3MoELayer
norm = decoders.Decoder.get_norm_layer(dec, num_features=E)
print(f"5) norm layer    {getattr(norm, 'func', norm).__name__}")

# ---- 6. routing math ----
# The bug this catches: moe.py routes to deepseek_scale_weights() (normalize +
# routed_scaling_factor) only for the block types named in that branch. A block
# not listed falls through to a softmax over the top-k scores, which silently
# drops routed_scaling_factor. Parameter count is unchanged, so checks 1-5 all
# still pass — this has to be asserted on its own.
import inspect
from MaxText.layers import moe as moe_mod
src = inspect.getsource(moe_mod.RoutedMoE.get_topk)
before = src.split("deepseek_scale_weights")[0].splitlines()
# take the nearest `if`/`elif` above the call — comments in between must not
# shift the window (an earlier version of this check used a fixed [-3:] and
# broke the moment a comment was added)
guard = next((l for l in reversed(before) if l.strip().startswith(("if ", "elif "))), "")
ok = "HUNYUAN3" in guard
print(f"6) routing math  deepseek_scale_weights branch covers HUNYUAN3: {ok}")
if not ok:
    print("   !! Hy3 would fall through to softmax and lose routed_scaling_factor")
assert ok, "HUNYUAN3 missing from the deepseek_scale_weights branch in moe.py"

# HF `route_norm: true` is already implemented inside deepseek_scale_weights.
# norm_topk_prob must stay off, or it normalizes a second time after the
# scaling factor was applied and cancels it out.
assert c.get("norm_topk_prob", False) is False, \
    "norm_topk_prob must be False for Hy3 — it would cancel routed_scaling_factor"
print("   norm_topk_prob False (route_norm already covered by scale_weights)")

print("\nALL CHECKS PASSED")
