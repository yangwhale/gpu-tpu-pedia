import time
import jax
from transformers import AutoModelForCausalLM, AutoTokenizer
import torchax 
from torchax.interop import torch_view

from jax.tree_util import register_pytree_node
from transformers import modeling_outputs

def output_flatten(v):
  return v.to_tuple(), None

def output_unflatten(aux, children):
  return modeling_outputs.CausalLMOutputWithPast(*children)

register_pytree_node(
  modeling_outputs.CausalLMOutputWithPast,
  output_flatten,
  output_unflatten,
)

from transformers import cache_utils

def _flatten_dynamic_cache(dynamic_cache):
  return (
      dynamic_cache.key_cache,
      dynamic_cache.value_cache,
  ), None

def _unflatten_dynamic_cache(aux, children):
  cache = cache_utils.DynamicCache(),
  cache.key_cache, cache.value_cache = children
  return cache

register_pytree_node(
  cache_utils.DynamicCache,
  _flatten_dynamic_cache,
  _unflatten_dynamic_cache,
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="jax")
print(model_inputs)

model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf", 
        torch_dtype="bfloat16", device_map="cpu")
        
weights, func = torchax.extract_jax(model)

def func_with_constant(weights, input_ids):
  res = func(weights, (input_ids, ), {'use_cache': False})
  return res

jitted_func = jax.jit(func_with_constant)
res = jitted_func(weights, model_inputs.input_ids)
print(res)