# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tencent Hunyuan 3 (Hy3, 295B-A21B) decoder layers.

Hy3 is a Qwen3-style attention stack bolted onto a DeepSeek-V3-style MoE:

  attention   GQA (64 heads / 8 KV groups, head_dim 128) + QK-LayerNorm,
              no QKV bias                       -> identical to Qwen3
  MoE         sigmoid routing + per-expert bias (aux-loss-free) +
              1 shared expert + routed_scaling  -> identical to DeepSeek V3
  layout      layer 0 dense, layers 1..79 MoE    -> same as DeepSeek V3's
              first_k_dense_replace

So neither `qwen3` nor `deepseek` alone can express it: the former has no
shared expert and no dense-first-layer support, the latter hardcodes MLA
attention. This module supplies only the wiring — both halves are imported
unchanged from the existing implementations:

  * `qwen3.self_attention_with_norm`   -> the GQA half
  * `moe.get_routed_and_shared_moe`    -> the DeepSeek MoE half
  * `deepseek.post_process`            -> shared epilogue

Layer classes are returned as a 2-element list (dense, moe) so that the
existing `first_num_dense_layers` scan machinery in `decoders.py` — written
for DeepSeek — drives Hy3 as well.
"""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from jax.sharding import Mesh
import jax.numpy as jnp

from flax import linen as nn

from MaxText.common_types import Config
from MaxText.layers import deepseek
from MaxText.layers import initializers
from MaxText.layers import linears
from MaxText.layers import moe
from MaxText.layers import qwen3
from MaxText.layers.quantizations import AqtQuantization as Quant
from MaxText.inference import page_manager


class Hunyuan3DenseLayer(nn.Module):
  """Hy3 dense layer — layer 0 only (`first_num_dense_layers=1`).

  GQA attention + a plain SwiGLU MLP of width `mlp_dim` (13312).
  """

  config: Config
  mesh: Mesh
  model_mode: str
  quant: None | Quant = None

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      page_state: None | page_manager.PageState = None,
      slot: None | int = None,
  ):
    cfg = self.config

    # `checkpoint_name(inputs, "decoder_layer_input")` happens inside this
    # helper — do not repeat it here or remat will see two identical names.
    hidden_states, residual_after_attention = qwen3.self_attention_with_norm(
        inputs,
        cfg,
        self.mesh,
        self.quant,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
    )

    mlp_output = linears.mlp_block(
        in_features=hidden_states.shape[-1],
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="mlp",
        config=cfg,
        quant=self.quant,
    )(hidden_states, deterministic=deterministic)

    layer_output = residual_after_attention + mlp_output
    layer_output = nn.with_logical_constraint(
        layer_output,
        ("activation_batch", "activation_length", "activation_embed"),
    )
    return deepseek.post_process(cfg, layer_output, self.sow)


class Hunyuan3MoELayer(nn.Module):
  """Hy3 MoE layer — layers 1..79.

  GQA attention + DeepSeek-V3 MoE block (192 routed experts, top-8, sigmoid
  routing with a learned per-expert bias, plus 1 shared expert).

  Uses `get_routed_and_shared_moe` rather than `get_routed_moe`: the latter
  returns a bare `RoutedMoE` and would silently drop Hy3's shared expert.
  """

  config: Config
  mesh: Mesh
  model_mode: str
  quant: None | Quant = None

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      page_state: None | page_manager.PageState = None,
      slot: None | int = None,
  ):
    cfg = self.config

    hidden_states, residual_after_attention = qwen3.self_attention_with_norm(
        inputs,
        cfg,
        self.mesh,
        self.quant,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
    )

    # `RoutedAndSharedMoE` reads num_experts / num_experts_per_tok /
    # moe_mlp_dim / shared_experts / routed_score_func / routed_bias /
    # routed_scaling_factor straight off the config, so unlike
    # `get_routed_moe` there is nothing to pass positionally here.
    mlp_output = moe.get_routed_and_shared_moe(
        name="Hunyuan3MoeBlock_0",
        config=cfg,
        mesh=self.mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        quant=self.quant,
    )(hidden_states)

    mlp_output = nn.with_logical_constraint(
        mlp_output, ("activation_batch", "activation_length", "activation_embed")
    )

    layer_output = residual_after_attention + mlp_output
    layer_output = nn.with_logical_constraint(
        layer_output,
        ("activation_batch", "activation_length", "activation_embed"),
    )
    return deepseek.post_process(cfg, layer_output, self.sow)
