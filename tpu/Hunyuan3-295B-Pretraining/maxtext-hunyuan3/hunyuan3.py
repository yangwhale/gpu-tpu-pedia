"""Hunyuan3 (Hy3, 295B-A21B) decoder layers.

Hy3 is DeepSeek-V3's MoE bolted onto Qwen3's attention:

  attention   GQA 64/8 + head_dim 128 + qk_norm, no MLA
              -> identical to Qwen3, so AttentionWithNorm is reused as-is
  MoE         sigmoid routing + per-expert bias + routed_scaling_factor,
              192 routed experts + 1 shared, first layer dense
              -> identical to DeepSeek V3, so RoutedAndSharedMoE is reused

Only the wiring is new. The two layer classes below differ from
Qwen3DecoderLayer / Qwen3MoeDecoderLayer in exactly one line each:
the dense layer is unchanged, and the MoE layer swaps RoutedMoE for
RoutedAndSharedMoE (RoutedMoE alone would silently drop the shared expert).
"""

from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax import linen as nn
from flax import nnx

from maxtext.common.common_types import Config
from maxtext.layers import initializers as max_initializers
from maxtext.layers import moe
from maxtext.layers import nnx_wrappers
from maxtext.layers.linears import MlpBlock
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.models.qwen3 import AttentionWithNorm


class Hunyuan3DenseLayer(AttentionWithNorm):
  """Hy3 dense layer — layer 0 only (first_num_dense_layers=1)."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
      quant: None | Quant = None,
      layer_idx: int = -1,
  ):
    # Callers are inconsistent: the nnx decoder builds DeepSeek-style layers
    # without passing quant, while the linen path always does. Keep quant and
    # layer_idx optional so both construction sites work.
    super().__init__(config, mesh, model_mode, quant, rngs)
    self.layer_idx = layer_idx
    self.mlp = MlpBlock(
        in_features=config.emb_dim,
        intermediate_dim=config.mlp_dim,
        activations=config.mlp_activations,
        intermediate_dropout_rate=config.dropout_rate,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        config=config,
        mesh=mesh,
        quant=quant,
        model_mode=model_mode,
        rngs=rngs,
    )

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache: None | jnp.ndarray = None,
      attention_metadata: None | dict[str, Any] = None,
  ):
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    hidden_states, intermediate_inputs, kv_cache = self.apply_attention_with_norm(
        inputs,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )
    mlp_lnx = self.mlp(hidden_states, deterministic=deterministic)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, self.activation_axis_names)
    layer_output = intermediate_inputs + mlp_lnx
    layer_output = nn.with_logical_constraint(layer_output, self.activation_axis_names)
    return layer_output, kv_cache


class Hunyuan3MoELayer(AttentionWithNorm):
  """Hy3 MoE layer — layers 1..79."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
      quant: None | Quant = None,
      layer_idx: int = -1,
  ):
    # Callers are inconsistent: the nnx decoder builds DeepSeek-style layers
    # without passing quant, while the linen path always does. Keep quant and
    # layer_idx optional so both construction sites work.
    super().__init__(config, mesh, model_mode, quant, rngs)
    self.layer_idx = layer_idx
    # 属性名必须与 port.py 打进 train.py 的 _MOE_BLOCK_ATTR 表一致：无梯度 bias
    # 更新是按名字去 state 里找 gate.bias 的。改这里就要同步改 port.py。
    self.Hunyuan3MoeBlock_0 = moe.RoutedAndSharedMoE(
        config=config,
        mesh=mesh,
        kernel_init=max_initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        quant=quant,
        rngs=rngs,
    )

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache: None | jnp.ndarray = None,
      attention_metadata: None | dict[str, Any] = None,
  ):
    is_scan_carry = False
    if isinstance(inputs, tuple) and len(inputs) == 3:
      hidden_states, stacked_kv_cache, layer_idx = inputs
      kv_cache = stacked_kv_cache[layer_idx]
      inputs = hidden_states
      is_scan_carry = True
    elif isinstance(inputs, tuple):
      inputs = inputs[0]
    if isinstance(inputs, tuple):
      inputs = inputs[0]

    hidden_states, intermediate_inputs, kv_cache = self.apply_attention_with_norm(
        inputs,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )

    mlp_lnx, load_balance_loss, moe_bias_updates = self.Hunyuan3MoeBlock_0(hidden_states)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, self.activation_axis_names)

    # Both of these must go out via `sow`, not by assigning an nnx.Intermediate
    # directly: the training loop reads them as `value[0]`, and only `sow`
    # produces the tuple that indexing expects.
    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.sow(nnx.Intermediate, "moe_lb_loss", load_balance_loss)

    # DeepSeek V3's auxiliary-loss-free load balancing (arXiv 2408.15664): the
    # MoE block returns a per-expert bias delta, and the training loop applies
    # it to `gate.bias` *outside* the optimizer, after the gradient step. Drop
    # this third return value and the mechanism goes silently inert — the
    # config still says `routed_bias_update_rate: 0.001`, the bias never moves,
    # and the experts collapse with nothing in the logs to say so.
    if (
        self.config.routed_bias
        and self.config.routed_bias_update_rate > 0.0
        and moe_bias_updates is not None
    ):
      self.sow(nnx.Intermediate, "moe_bias_updates", moe_bias_updates)

    layer_output = intermediate_inputs + mlp_lnx
    layer_output = nn.with_logical_constraint(layer_output, self.activation_axis_names)

    if is_scan_carry:
      def update_cache(cache, val):
        if jnp.size(val) > 0:
          return cache.at[layer_idx].set(val)
        return cache

      stacked_kv_cache = jax.tree_util.tree_map(update_cache, stacked_kv_cache, kv_cache)
      return (layer_output, stacked_kv_cache, layer_idx + 1), None
    return layer_output, kv_cache


Hunyuan3DenseLayerToLinen = nnx_wrappers.to_linen_class(
    Hunyuan3DenseLayer,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

Hunyuan3MoELayerToLinen = nnx_wrappers.to_linen_class(
    Hunyuan3MoELayer,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)
