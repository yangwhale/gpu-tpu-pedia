#!/usr/bin/env python3
"""把 hunyuan3 补丁移植到新版 MaxText（src/maxtext，nnx 布局）。

老版补丁针对 src/MaxText（linen）。新版上游做了三件事让补丁变小：
  * layer 代码搬到 models/，并且从 linen 迁到 nnx
  * float32_gate_logits 取代了我自己加的 moe_router_dtype
  * routed_bias_update_rate 已经实现了无辅助损失的偏置更新
所以这里只剩「把 hunyuan3 加进各处按模型名列举的分支」。
"""
import re, sys, os

ROOT = os.environ.get("MAXTEXT_ROOT", "/tmp/mt-v7/src/maxtext")
if not os.path.isdir(ROOT):
  sys.exit(f"找不到 {ROOT} —— 用 MAXTEXT_ROOT=/path/to/maxtext/src/maxtext 指定")
changed = []


def edit(rel, fn, expect):
  """expect = 改完之后这个文件里应该出现多少处 HUNYUAN3 / hunyuan3。

  之前这里只断言「文件变了」，结果 maxtext_utils.py 三处改动里有一处的正则
  没匹配上（新版把单行 tuple 拆成了多行），静默漏掉，一直到复现审计才发现。
  按处数断言，漏一处就当场炸。
  """
  p = os.path.join(ROOT, rel)
  s0 = open(p).read()
  s = fn(s0)
  n = s.upper().count("HUNYUAN3")
  assert n == expect, f"{rel}: 命中 {n} 处，期望 {expect} 处——锚点可能变了"
  open(p, "w").write(s)
  changed.append(f"{rel}({n})")


# 1) 枚举
def _ct(s):
  return s.replace('  QWEN3_MOE = "qwen3_moe"',
                   '  QWEN3_MOE = "qwen3_moe"\n  HUNYUAN3 = "hunyuan3"', 1)
edit("common/common_types.py", _ct, expect=2)


# 2) decoders：import + 分派 + 所有 DEEPSEEK 等值判断
def _dec(s):
  s = s.replace("from maxtext.models import deepseek\n",
                "from maxtext.models import deepseek\nfrom maxtext.models import hunyuan3\n", 1)
  s = s.replace(
      """      case DecoderBlockType.DEEPSEEK:
        return [
            deepseek.DeepSeekDenseLayerToLinen,
            deepseek.DeepSeekMoELayerToLinen,
        ]""",
      """      case DecoderBlockType.DEEPSEEK:
        return [
            deepseek.DeepSeekDenseLayerToLinen,
            deepseek.DeepSeekMoELayerToLinen,
        ]
      case DecoderBlockType.HUNYUAN3:
        # Must be a 2-element [dense, moe] list: the first_num_dense_layers
        # scan machinery below indexes it positionally.
        return [
            hunyuan3.Hunyuan3DenseLayerToLinen,
            hunyuan3.Hunyuan3MoELayerToLinen,
        ]""", 1)
  # 走 DeepSeek 那条 dense+moe 混合 scan 路径
  s = re.sub(r"== DecoderBlockType\.DEEPSEEK(?![0-9A-Z_])",
             "in (DecoderBlockType.DEEPSEEK, DecoderBlockType.HUNYUAN3)", s)
  # 支持 rms_norm / 标准 decoder 的模型清单
  s = s.replace("        DecoderBlockType.DEEPSEEK,\n        DecoderBlockType.DEEPSEEK4,",
                "        DecoderBlockType.DEEPSEEK,\n        DecoderBlockType.DEEPSEEK4,\n        DecoderBlockType.HUNYUAN3,")
  return s
edit("layers/decoders.py", _dec, expect=11)


# 3) moe：路由缩放分支 + 五处 model_name 门
def _moe(s):
  s = s.replace(
      "if self.config.decoder_block in (ctypes.DecoderBlockType.DEEPSEEK, ctypes.DecoderBlockType.DEEPSEEK4):\n"
      "      top_k_weights = self.deepseek_scale_weights(top_k_weights)",
      "if self.config.decoder_block in (\n"
      "        ctypes.DecoderBlockType.DEEPSEEK,\n"
      "        ctypes.DecoderBlockType.DEEPSEEK4,\n"
      "        ctypes.DecoderBlockType.HUNYUAN3,\n"
      "    ):\n"
      "      # Hy3 shares DeepSeek's sigmoid routing. Falling through to the\n"
      "      # softmax branch would silently drop routed_scaling_factor (2.826).\n"
      "      top_k_weights = self.deepseek_scale_weights(top_k_weights)", 1)
  # 这些门决定 top-k 用「加了 bias 的分数」选、但取「不带 bias 的权重值」
  n = len(re.findall(r'startswith\(\("deepseek3", "deepseek4"\)\)', s))
  assert n == 5, f"model_name 门的数量变了: {n}"
  s = s.replace('startswith(("deepseek3", "deepseek4"))',
                'startswith(("deepseek3", "deepseek4", "hunyuan3"))')
  return s
edit("layers/moe.py", _moe, expect=6)


# 4) FLOP 口径三处（详见 README 的 bug #5）
def _utils(s):
  # 这一处上游写成了多行 tuple，早先的单行正则匹配不到，必须按多行的实际文本改
  old_ffn = ("        DecoderBlockType.GEMMA4,\n        DecoderBlockType.DEEPSEEK4,\n    ):\n"
             "      total_ffn_flops = calculate_routed_and_shared_ffn_tflops_per_device(config)")
  new_ffn = ("        DecoderBlockType.GEMMA4,\n        DecoderBlockType.DEEPSEEK4,\n"
             "        DecoderBlockType.HUNYUAN3,\n    ):\n"
             "      # Hy3 has DeepSeek's routed + shared + leading-dense structure. The\n"
             "      # generic branch below sizes the experts with mlp_dim (13312, the dense\n"
             "      # width) instead of moe_mlp_dim (1536) and skips the shared expert,\n"
             "      # inflating reported TFLOP/s ~5x. Training is unaffected; MFU is not.\n"
             "      total_ffn_flops = calculate_routed_and_shared_ffn_tflops_per_device(config)")
  assert old_ffn in s, "FLOP 的 MoE FFN 分支锚点变了"
  s = s.replace(old_ffn, new_ffn, 1)
  s = s.replace("  if config.decoder_block == DecoderBlockType.DEEPSEEK:\n    num_dense_layers = config.first_num_dense_layers",
                "  if config.decoder_block in (DecoderBlockType.DEEPSEEK, DecoderBlockType.HUNYUAN3):\n    num_dense_layers = config.first_num_dense_layers", 1)
  s = s.replace("  elif config.decoder_block == DecoderBlockType.DEEPSEEK:\n    learnable_weight_tflops = (",
                "  elif config.decoder_block in (DecoderBlockType.DEEPSEEK, DecoderBlockType.HUNYUAN3):\n"
                "    # total_ffn_flops is already summed over layers by the helper; the\n"
                "    # generic branch would multiply by num_decoder_layers a second time.\n"
                "    learnable_weight_tflops = (", 1)
  # 最后一层 MoE 的 FLOP 重建：名单里没有 Hy3 就不算共享专家那部分，报表偏低。
  a = ("      if config.decoder_block in (\n          DecoderBlockType.DEEPSEEK,\n"
       "          DecoderBlockType.LLAMA4,\n          DecoderBlockType.QWEN3_NEXT,\n"
       "          DecoderBlockType.GEMMA4,\n      ):\n        shared_flops = (")
  b = ("      if config.decoder_block in (\n          DecoderBlockType.DEEPSEEK,\n"
       "          DecoderBlockType.HUNYUAN3,\n          DecoderBlockType.LLAMA4,\n"
       "          DecoderBlockType.QWEN3_NEXT,\n          DecoderBlockType.GEMMA4,\n      ):\n        shared_flops = (")
  assert s.count(a) == 1, "maxtext_utils.py: shared_flops 名单锚点没命中"
  s = s.replace(a, b, 1)

  return s
edit("utils/maxtext_utils.py", _utils, expect=4)


# 5) pydantic 的 model_name 白名单（新版用 Literal 取代了旧版的 validate_model_name）
def _types(s):
  # 两个都要加：只加 295b 的话仓库里发的 hunyuan3-smoke.yml 用不了，
  # 而文档恰恰让人先用冒烟配置——2026-07-29 分支验证时撞到。
  s = s.replace('    "deepseek3-671b",\n',
                '    "deepseek3-671b",\n    "hunyuan3-295b",\n    "hunyuan3-smoke",\n', 1)
  # loss-free 负载均衡的 validator 也是按 decoder block 名字列举的
  old = ('      if self.routed_bias and self.routed_bias_update_rate > 0.0 '
         'and self.decoder_block != DecoderBlockType.DEEPSEEK:\n'
         '        raise ValueError("Loss-free load balancing is only supported for the DeepSeek decoder block.")')
  new = ('      if (\n'
         '          self.routed_bias\n'
         '          and self.routed_bias_update_rate > 0.0\n'
         '          and self.decoder_block not in (DecoderBlockType.DEEPSEEK, DecoderBlockType.HUNYUAN3)\n'
         '      ):\n'
         '        # Hy3 uses the same aux-loss-free scheme as DSV3: the per-expert bias\n'
         '        # shifts top-k selection only, updated by a non-gradient rule.\n'
         '        raise ValueError("Loss-free load balancing is only supported for the DeepSeek decoder block.")')
  assert old in s
  s = s.replace(old, new, 1)
  # 流水线并行层数推导：DeepSeek 走「总层数 - 稠密首层」，Hy3 同构，
  # 落到 else 会把稠密首层也算进 MoE 段，开 PP 之后分配差一层。
  a = "      if self.decoder_block == DecoderBlockType.DEEPSEEK:\n        moe_layers ="
  b = ("      if self.decoder_block in (DecoderBlockType.DEEPSEEK, DecoderBlockType.HUNYUAN3):\n"
       "        moe_layers =")
  assert s.count(a) == 1, "types.py: pipeline_parallel_layers 锚点没命中"
  s = s.replace(a, b, 1)
  # muon 优化器白名单
  a = "    if self.opt_type == \"muon\" and self.decoder_block not in [\n        DecoderBlockType.DEEPSEEK,\n"
  b = ("    if self.opt_type == \"muon\" and self.decoder_block not in [\n"
       "        DecoderBlockType.DEEPSEEK,\n        DecoderBlockType.HUNYUAN3,\n")
  assert s.count(a) == 1, "types.py: muon 白名单锚点没命中"
  s = s.replace(a, b, 1)
  return s


edit("configs/types.py", _types, expect=5)


# 6) nnx_decoders：第三张分派表 + rms_norm 白名单 + dense/moe 混合 scan 判定
#    这一张最容易漏——前两张在 decoders.py，这张在另一个文件里，
#    漏了会报 "Incorrect decoder_block name"，看不出是漏了哪张表。
def _nnxdec(s):
  s = s.replace("from maxtext.models import (\n    deepseek,\n",
                "from maxtext.models import (\n    deepseek,\n    hunyuan3,\n", 1)
  s = s.replace("        DecoderBlockType.DEEPSEEK: get_deepseek(),\n",
                "        DecoderBlockType.DEEPSEEK: get_deepseek(),\n"
                "        DecoderBlockType.HUNYUAN3: [hunyuan3.Hunyuan3DenseLayer, hunyuan3.Hunyuan3MoELayer],\n", 1)
  s = s.replace("        DecoderBlockType.MIXTRAL,\n        DecoderBlockType.DEEPSEEK,\n        DecoderBlockType.GEMMA,\n",
                "        DecoderBlockType.MIXTRAL,\n        DecoderBlockType.DEEPSEEK,\n"
                "        DecoderBlockType.HUNYUAN3,\n        DecoderBlockType.GEMMA,\n", 1)
  s = s.replace("    self.is_deepseek = self.config.decoder_block == DecoderBlockType.DEEPSEEK",
                "    self.is_deepseek = self.config.decoder_block in "
                "(DecoderBlockType.DEEPSEEK, DecoderBlockType.HUNYUAN3)", 1)

  return s
edit("layers/nnx_decoders.py", _nnxdec, expect=8)


# 7) 训练主循环：无梯度 bias 更新的路径把 DeepSeek 的模块属性名写死了，两处。
#    不改的话 routed_bias_update_rate 配了也没用 —— 会直接 AttributeError（见 README §八 bug #9）。
#    这里不能沿用 edit()，因为 train.py 里出现的是 "Hunyuan3MoeBlock_0" 而非 "hunyuan3"。
def _train(s):
  a1 = '("params", "decoder", "moe_layers", "DeepSeekMoeBlock_0", "MoeBlock_0", "gate", "bias")'
  b1 = '("params", "decoder", "moe_layers", _moe_block_attr(config), "MoeBlock_0", "gate", "bias")'
  a2 = "new_state.model.decoder.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias"
  b2 = "getattr(new_state.model.decoder.moe_layers, _moe_block_attr(config)).MoeBlock_0.gate.bias"
  for a in (a1, a2):
    assert s.count(a) == 1, f"train.py: 锚点命中 {s.count(a)} 次，期望 1 次 —— 上游可能改过"
  s = s.replace(a1, b1).replace(a2, b2)
  helper = (
      '_MOE_BLOCK_ATTR = {"deepseek": "DeepSeekMoeBlock_0", "hunyuan3": "Hunyuan3MoeBlock_0"}\n\n\n'
      "def _moe_block_attr(config):\n"
      '  key = getattr(config.decoder_block, "value", config.decoder_block)\n'
      "  return _MOE_BLOCK_ATTR[str(key)]\n\n\n"
  )
  m = re.search(r"^def ", s, re.M)
  assert m, "train.py: 找不到模块级 def，无法插入 helper"
  return s[:m.start()] + helper + s[m.start():]


# ── 8) 以下 5 个文件不是「注册」，而是**结构判断**：它们按「dense 段 + MoE 段」
#    两段式来分组，Hy3 与 DeepSeek 结构相同，落到 else 的单段式分支会算错。
#    这些之前没改是因为「我们没开那个开关」——那不是理由，开了就错，提前备好。
def _two_group(rel, expect_hits):
  """把 `== DecoderBlockType.DEEPSEEK` 放宽成 `in (DEEPSEEK, HUNYUAN3)`。"""
  def fn(s):
    a = "if config.decoder_block == DecoderBlockType.DEEPSEEK:"
    b = "if config.decoder_block in (DecoderBlockType.DEEPSEEK, DecoderBlockType.HUNYUAN3):"
    n = s.count(a)
    assert n == expect_hits, f"{rel}: 命中 {n} 处，期望 {expect_hits} 处"
    return s.replace(a, b)
  edit(rel, fn, expect=expect_hits)


# 权重导出：分别展开 dense_layers / moe_layers。走 else 只会展开一组名叫 layers 的，
# 做 SFT 导权重时结构对不上。三处同样的写法。
_two_group("utils/generate_param_only_checkpoint.py", 3)
# RL 的参数同步，同样按两段分组
_two_group("experimental/rl/grpo_utils.py", 1)


# MlpBlock 自己的 norm 白名单——**else 分支是 raise ValueError**，
# 只要 use_pre_norm=True 就崩在「Incorrect decoder_block name」。
def _linears(s):
  a = "        DecoderBlockType.QWEN3,\n        DecoderBlockType.DEEPSEEK,\n        DecoderBlockType.LLAMA4,\n"
  assert s.count(a) == 1, "linears.py: norm 白名单锚点没命中"
  return s.replace(
      a,
      "        DecoderBlockType.QWEN3,\n        DecoderBlockType.DEEPSEEK,\n"
      "        DecoderBlockType.HUNYUAN3,\n        DecoderBlockType.LLAMA4,\n", 1)
edit("layers/linears.py", _linears, expect=1)


# MTP + batch split 的重分片路径
def _mtp(s):
  a = "if self.config.decoder_block == DecoderBlockType.DEEPSEEK and self.config.use_batch_split_schedule:"
  b = ("if (\n          self.config.decoder_block in (DecoderBlockType.DEEPSEEK, DecoderBlockType.HUNYUAN3)\n"
       "          and self.config.use_batch_split_schedule\n      ):")
  assert s.count(a) == 1, "multi_token_prediction.py: 锚点没命中"
  return s.replace(a, b, 1)
edit("layers/multi_token_prediction.py", _mtp, expect=1)


# linen 的逐层量化断言
def _lwq(s):
  a = "assert config.decoder_block == common_types.DecoderBlockType.DEEPSEEK, ("
  b = ("assert config.decoder_block in (\n      common_types.DecoderBlockType.DEEPSEEK,\n"
       "      common_types.DecoderBlockType.HUNYUAN3,\n  ), (")
  assert s.count(a) == 1, "layerwise_quantization.py: 锚点没命中"
  return s.replace(a, b, 1)
edit("utils/layerwise_quantization.py", _lwq, expect=1)


_p = os.path.join(ROOT, "trainers/pre_train/train.py")
_s0 = open(_p).read()
_s = _train(_s0)
assert _s.count("Hunyuan3MoeBlock_0") == 1 and _s.count("_moe_block_attr") == 3, "train.py 补丁不完整"
compile(_s, _p, "exec")
open(_p, "w").write(_s)
changed.append("trainers/pre_train/train.py")

print("已改:", ", ".join(changed))
