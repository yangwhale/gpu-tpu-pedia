import time
import jax
from transformers import AutoModelForCausalLM, AutoTokenizer
import torchax 
from torchax.interop import torch_view
from jax.tree_util import register_pytree_node
from transformers import modeling_outputs, cache_utils


def setup_pytree_registrations():
    """注册必要的pytree节点以支持JAX转换"""
    
    # 注册CausalLMOutputWithPast
    def output_flatten(v):
        return v.to_tuple(), None

    def output_unflatten(aux, children):
        return modeling_outputs.CausalLMOutputWithPast(*children)

    register_pytree_node(
        modeling_outputs.CausalLMOutputWithPast,
        output_flatten,
        output_unflatten,
    )

    # 注册DynamicCache
    def _flatten_dynamic_cache(dynamic_cache):
        return (
            dynamic_cache.key_cache,
            dynamic_cache.value_cache,
        ), None

    def _unflatten_dynamic_cache(aux, children):
        cache = cache_utils.DynamicCache()
        cache.key_cache, cache.value_cache = children
        return cache

    register_pytree_node(
        cache_utils.DynamicCache,
        _flatten_dynamic_cache,
        _unflatten_dynamic_cache,
    )


def load_model_and_tokenizer(model_name="meta-llama/Llama-2-7b-hf"):
    """加载模型和tokenizer"""
    print(f"正在加载模型: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype="bfloat16", 
        device_map="cpu"
    )
    
    return model, tokenizer


def prepare_jax_function(model):
    """提取JAX函数和权重"""
    weights, func = torchax.extract_jax(model)
    
    def func_with_constant(weights, input_ids):
        res = func(weights, (input_ids,), {'use_cache': False})
        return res
    
    return weights, func_with_constant


def run_inference_benchmark(func, weights, input_ids, num_iterations=3):
    """运行推理基准测试"""
    print(f"\n运行{num_iterations}次推理基准测试...")
    
    for i in range(num_iterations):
        start = time.time()
        res = func(weights, input_ids)
        jax.block_until_ready(res)
        end = time.time()
        print(f"迭代 {i}: {end - start:.4f} 秒")
    
    return res


def main():
    """主函数"""
    print("=" * 60)
    print("JAX + Hugging Face 模型推理示例")
    print("=" * 60)
    
    # 1. 设置pytree注册
    setup_pytree_registrations()
    
    # 2. 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer("meta-llama/Llama-2-7b-hf")
    
    # 3. 准备输入
    print("\n准备输入文本...")
    model_inputs = tokenizer(
        ["The secret to baking a good cake is "], 
        return_tensors="jax"
    )
    print(f"输入tokens: {model_inputs}")
    
    # 4. 提取JAX函数
    print("\n提取JAX函数...")
    weights, func_with_constant = prepare_jax_function(model)
    
    # 5. 首次运行（未JIT）
    print("\n首次运行（未JIT编译）...")
    # res = func_with_constant(weights, model_inputs.input_ids)
    res = run_inference_benchmark(
        func_with_constant, 
        weights, 
        model_inputs.input_ids, 
        num_iterations=3
    )
    print(f"输出shape: {res[0].shape if hasattr(res, '__getitem__') else 'N/A'}")
    
    # 6. JIT编译并运行基准测试
    print("\nJIT编译函数...")
    jitted_func = jax.jit(func_with_constant)
    
    res = run_inference_benchmark(
        jitted_func, 
        weights, 
        model_inputs.input_ids, 
        num_iterations=3
    )
    print(f"输出shape: {res[0].shape if hasattr(res, '__getitem__') else 'N/A'}")
    
    print("\n" + "=" * 60)
    print("推理完成！")
    print("=" * 60)
    
    # 注释掉的generate代码（备用）
    # env = torchax.default_env()
    # with env:
    #     model.to('jax')
    #     model.model.rotary_emb.original_inv_freq = model.model.rotary_emb.original_inv_freq.to('jax')
    #     jmodel = torchax.interop.JittableModule(model)
    #     model_inputs = dict(model_inputs)
    #     generated_ids = jmodel.generate(**torch_view(model_inputs))
    # print(generated_ids)
    # print(tokenizer.batch_decode(generated_ids.cpu())[0])


if __name__ == "__main__":
    main()
