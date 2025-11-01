import time
import jax
from transformers import AutoModelForCausalLM, AutoTokenizer
import torchax 
from torchax.interop import torch_view
from jax.tree_util import register_pytree_node
from transformers import modeling_outputs, cache_utils
from jax.sharding import PartitionSpec as P, NamedSharding


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


def shard_weights_llama(mesh, weights):
    """
    对Llama模型的权重进行分片
    
    根据不同的权重矩阵类型采用不同的分片策略：
    - q_proj, k_proj, v_proj, gate_proj, up_proj: 在第一维分片
    - o_proj, down_proj, lm_head, embed_tokens: 在第二维分片
    - 其他权重: 复制到所有设备
    
    Args:
        mesh: JAX设备网格
        weights: 模型权重字典
        
    Returns:
        分片后的权重字典
    """
    result = {}
    for k, v in weights.items():
        if (('q_proj' in k) or 
            ('k_proj' in k) or 
            ('v_proj' in k) or 
            ('gate_proj' in k) or 
            ('up_proj' in k)):
            # 注意力和前馈网络的投影层在第一维分片
            sharding = P('axis', None)
        elif(('o_proj' in k) or 
             ('down_proj' in k) or 
             ('lm_head.weight' in k) or 
             ('embed_tokens' in k)):
            # 输出投影和词嵌入在第二维分片
            sharding = P(None, 'axis')
        else:
            # 其他权重（如LayerNorm）复制到所有设备
            sharding = P()  # replicated
        result[k] = jax.device_put(v, NamedSharding(mesh, sharding))
    return result


def prepare_jax_function(model):
    """提取JAX函数和权重"""
    print("提取JAX函数和权重...")
    weights, func = torchax.extract_jax(model)
    
    def func_with_constant(weights, input_ids):
        """包装函数，固定use_cache参数为False"""
        res = func(weights, (input_ids,), {'use_cache': False})
        return res
    
    return weights, func_with_constant


def setup_mesh_and_sharding(weights, input_ids):
    """
    创建设备网格并对权重和输入进行分片
    
    Args:
        weights: 模型权重
        input_ids: 输入token IDs
        
    Returns:
        mesh: 设备网格
        sharded_weights: 分片后的权重
        sharded_input_ids: 分片后的输入
    """
    print(f"\n创建设备网格（设备数: {jax.device_count()}）...")
    mesh = jax.make_mesh((jax.device_count(),), ('axis',))
    
    print("对模型权重进行分片...")
    sharded_weights = shard_weights_llama(mesh, weights)
    
    print("对输入数据进行分片（复制模式）...")
    sharded_input_ids = jax.device_put(
        input_ids, 
        NamedSharding(mesh, P())  # 复制到所有设备
    )
    
    # 确保所有分片操作完成
    jax.block_until_ready(sharded_weights)
    
    return mesh, sharded_weights, sharded_input_ids


def run_inference_benchmark(func, weights, input_ids, num_iterations=3, enable_profiler=False):
    """
    运行推理基准测试
    
    Args:
        func: 推理函数
        weights: 模型权重
        input_ids: 输入token IDs
        num_iterations: 迭代次数
        enable_profiler: 是否启用profiler
        
    Returns:
        最后一次推理的结果
    """
    print(f"\n运行{num_iterations}次推理基准测试...")
    
    if enable_profiler:
        print("启用JAX profiler...")
        profiler_context = jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=False)
    else:
        # 创建一个空的上下文管理器
        from contextlib import nullcontext
        profiler_context = nullcontext()
    
    with profiler_context:
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
    print("JAX + Hugging Face 模型推理示例（带分片支持）")
    print("=" * 60)
    
    # 1. 设置pytree注册
    print("\n1. 设置pytree注册...")
    setup_pytree_registrations()
    
    # 2. 加载模型和tokenizer
    print("\n2. 加载模型和tokenizer...")
    model, tokenizer = load_model_and_tokenizer("meta-llama/Llama-2-7b-hf")
    
    # 3. 准备输入
    print("\n3. 准备输入文本...")
    model_inputs = tokenizer(
        ["The secret to baking a good cake is "], 
        return_tensors="jax"
    )
    print(f"输入tokens: {model_inputs}")
    
    # 4. 提取JAX函数
    print("\n4. 提取JAX函数...")
    weights, func_with_constant = prepare_jax_function(model)
    
    # 5. 首次运行（未JIT，未分片）
    print("\n" + "=" * 60)
    print("5. 首次运行（未JIT编译，未分片）")
    print("=" * 60)
    res = run_inference_benchmark(
        func_with_constant, 
        weights, 
        model_inputs.input_ids, 
        num_iterations=3
    )
    print(f"输出shape: {res[0].shape if hasattr(res, '__getitem__') else 'N/A'}")
    
    # 6. JIT编译（未分片）
    print("\n" + "=" * 60)
    print("6. JIT编译运行（未分片）")
    print("=" * 60)
    print("JIT编译函数...")
    jitted_func = jax.jit(func_with_constant)
    
    res = run_inference_benchmark(
        jitted_func, 
        weights, 
        model_inputs.input_ids, 
        num_iterations=3
    )
    print(f"输出shape: {res[0].shape if hasattr(res, '__getitem__') else 'N/A'}")
    
    # 7. 设置分片并运行
    print("\n" + "=" * 60)
    print("7. 分片模式运行（JIT编译 + 权重分片）")
    print("=" * 60)
    mesh, sharded_weights, sharded_input_ids = setup_mesh_and_sharding(
        weights, 
        model_inputs.input_ids
    )
    
    res = run_inference_benchmark(
        jitted_func, 
        sharded_weights, 
        sharded_input_ids, 
        num_iterations=3,
        enable_profiler=True
    )
    print(f"输出shape: {res[0].shape if hasattr(res, '__getitem__') else 'N/A'}")
    
    print("\n" + "=" * 60)
    print("推理完成！")
    print("=" * 60)
    print("\n性能总结：")
    print("- 未JIT编译: 最慢，每次都重新编译")
    print("- JIT编译: 更快，编译后缓存")
    print("- JIT + 分片: 在多设备环境下性能最优")
    print("- Profiler数据已保存到: /tmp/jax-trace")
    
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
