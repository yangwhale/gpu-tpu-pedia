import time
import jax
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
import torchax as tx
from torchax.interop import torch_view
from jax.sharding import PartitionSpec as P, NamedSharding
from jax.tree_util import register_pytree_node
from transformers import modeling_outputs, cache_utils
import torch


def setup_pytree_registrations():
    """
    注册必要的pytree节点以支持JAX转换
    
    包括：
    1. CausalLMOutputWithPast - 模型输出
    2. DynamicCache - 动态KV缓存
    3. StaticCache - 静态KV缓存（用于优化推理）
    """
    
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

    # 注册StaticCache（用于静态形状优化）
    def _flatten_static_cache(cache):
        return (
            cache.key_cache,
            cache.value_cache,
        ), (cache._config, cache.max_batch_size, cache.max_cache_len)

    def _unflatten_static_cache(aux, children):
        cache = cache_utils.StaticCache(*aux)
        cache._config = aux[0]
        cache.key_cache, cache.value_cache = children
        return cache

    register_pytree_node(
        cache_utils.StaticCache,
        _flatten_static_cache,
        _unflatten_static_cache,
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
    
    与jax_hg_02.py不同，这里使用apply_jax方法在torchax环境中进行分片
    
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

        # apply_jax 在包含 jax.Array 的 tensor 上调用 JAX 函数
        result[k] = v.apply_jax(jax.device_put, NamedSharding(mesh, sharding))
    return result


def run_twice_and_print_cache(model, input_ids):
    """
    运行两次前向传播并打印KV cache信息（使用动态缓存）
    
    演示：
    1. 第一次前向传播生成初始KV cache
    2. 获取下一个token
    3. 第二次前向传播使用已有的KV cache
    
    Args:
        model: 语言模型
        input_ids: 输入token IDs
    """
    print("\n使用动态缓存运行两次前向传播...")
    
    # 第一次前向传播
    res = model(input_ids)
    print(f'KV cache层数: {len(res[1])}')
    for k, v in res[1]:
        print(f'第一次KV cache shape: key={k.shape}, value={v.shape}')
        break

    # 获取下一个token（argmax预测）
    next_token = torch.argmax(res[0][:, -1], dim=-1)

    # 第二次前向传播，复用KV cache
    res = model(next_token.unsqueeze(0), past_key_values=res[1])
    print(f'KV cache层数: {len(res[1])}')
    for k, v in res[1]:
        print(f'第二次KV cache shape: key={k.shape}, value={v.shape}')
        break


def run_twice_and_print_cache_static(model, input_ids):
    """
    运行两次前向传播并打印KV cache信息（使用静态缓存）
    
    静态缓存预先分配固定大小的内存，避免动态分配开销，
    适合JIT编译优化
    
    Args:
        model: 语言模型
        input_ids: 输入token IDs
    """
    print("\n使用静态缓存运行两次前向传播...")
    
    # 创建静态缓存（预分配内存）
    past_key_values = StaticCache(
        config=model.config, 
        max_batch_size=1, 
        max_cache_len=50, 
        device='jax', 
        dtype=model.dtype
    )
    
    # 第一次前向传播
    res = model(input_ids, past_key_values=past_key_values)
    print(f'KV cache层数: {len(res[1].key_cache)}')
    if len(res[1].key_cache) > 0:
        k = res[1].key_cache[0]
        v = res[1].value_cache[0]
        print(f'第一次静态KV cache shape: key={k.shape}, value={v.shape}')

    # 获取下一个token
    next_token = torch.argmax(res[0][:, -1], dim=-1)

    # 第二次前向传播
    res = model(next_token.unsqueeze(0), past_key_values=res[1])
    print(f'KV cache层数: {len(res[1].key_cache)}')
    if len(res[1].key_cache) > 0:
        k = res[1].key_cache[0]
        v = res[1].value_cache[0]
        print(f'第二次静态KV cache shape: key={k.shape}, value={v.shape}')


def autoregressive_decode(model, input_ids, tokenizer, max_tokens=50):
    """
    自回归解码生成文本（使用动态缓存）
    
    逐个token生成，每次使用之前的KV cache加速
    
    Args:
        model: 语言模型
        input_ids: 初始输入token IDs
        tokenizer: 分词器
        max_tokens: 最大生成token数
        
    Returns:
        生成的token列表
    """
    print("\n开始自回归解码（动态缓存）...")
    start = time.perf_counter()
    
    # 初始前向传播
    res = model(input_ids)
    next_token = torch.argmax(res[0][:, -1], dim=-1)
    result_tokens = [int(next_token.item())]

    # 逐个生成token
    for _ in range(max_tokens):
        res = model(next_token.unsqueeze(0), past_key_values=res[1])
        next_token = torch.argmax(res[0][:, -1], dim=-1)
        if next_token.item() == tokenizer.eos_token:
            break
        result_tokens.append(next_token.item())
    
    end = time.perf_counter()
    
    decoded_text = tokenizer.batch_decode([result_tokens])
    print(f'生成文本: {decoded_text}')
    print(f'耗时: {end - start:.4f} 秒')
    
    return result_tokens


def autoregressive_decode_static(model, input_ids, tokenizer, mesh, max_tokens=50):
    """
    自回归解码生成文本（使用静态缓存 + JIT编译）
    
    优化特性：
    1. 静态缓存 - 预分配固定大小内存
    2. JIT编译 - 加速单token解码
    3. KV cache分片 - 在多设备上并行
    
    参考：https://huggingface.co/docs/transformers/v4.44.0/en/llm_optims?static-kv=advanced+usage%3A+control+Static+Cache#static-kv-cache-and-torchcompile
    
    Args:
        model: 语言模型
        input_ids: 初始输入token IDs
        tokenizer: 分词器
        mesh: JAX设备网格
        max_tokens: 最大生成token数
    """
    print("\n开始自回归解码（静态缓存 + JIT）...")
    
    def decode_one_token(model_weights, cur_token, input_pos, cache_position, past_key_values):
        """
        解码单个token的函数（将被JIT编译）
        
        使用torch.func.functional_call进行无状态调用
        """
        logits, cache = torch.func.functional_call(
            model, 
            model_weights,  # 权重字典
            (cur_token,),   # 位置参数
            dict(
                position_ids=input_pos,
                cache_position=cache_position,
                past_key_values=past_key_values,
                return_dict=False,
                use_cache=True
            )  # 关键字参数
        )
        new_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
        return new_token, cache

    # JIT编译解码函数
    jitted = tx.interop.jax_jit(decode_one_token)

    batch_size, seq_length = input_ids.shape
    model_weights = model.state_dict()
    
    with torch.no_grad():
        start = time.perf_counter()
        
        # 创建静态缓存
        past_key_values = StaticCache(
            config=model.config, 
            max_batch_size=1, 
            max_cache_len=max_tokens, 
            device='jax', 
            dtype=model.dtype
        )
        past_key_values._config = model.config
        cache_position = torch.arange(seq_length, device='jax')
        generated_ids = []

        # 首次前向传播（prefill阶段）
        logits, past_key_values = model(
            input_ids, 
            cache_position=cache_position, 
            past_key_values=past_key_values, 
            return_dict=False, 
            use_cache=True
        )
        next_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
        generated_ids.append(next_token[:, 0].item())

        # 对KV cache进行分片（在num_heads维度分片）
        for k in past_key_values.key_cache:
            k.apply_jax_(jax.device_put, NamedSharding(mesh, P(None, 'axis', None, None)))
        for v in past_key_values.value_cache:
            v.apply_jax_(jax.device_put, NamedSharding(mesh, P(None, 'axis', None, None)))

        cache_position = torch.tensor([seq_length + 1], device='jax')
        
        # 逐个生成token（decode阶段）
        for i in range(1, max_tokens):
            next_token, past_key_values = jitted(
                model_weights,
                next_token.clone(), 
                None, 
                cache_position, 
                past_key_values
            )
            generated_ids.append(next_token.int().item())
            cache_position += 1
        
        end = time.perf_counter()

    # 解码生成的文本
    text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(f'生成文本: {text}')
    print(f'耗时: {end - start:.4f} 秒')


def main():
    """主函数"""
    print("=" * 60)
    print("JAX + Hugging Face 自回归解码示例")
    print("（StaticCache + JIT优化）")
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
        return_tensors="pt"
    )
    print(f"输入tokens: {model_inputs}")
    
    # 4. 创建设备网格
    print(f"\n4. 创建设备网格（设备数: {jax.device_count()}）...")
    mesh = jax.make_mesh((jax.device_count(),), ('axis',))
    
    # 5. 在torchax环境中进行模型转换和分片
    print("\n5. 转换模型到JAX并进行权重分片...")
    env = tx.default_env()
    
    with env:
        # 将模型移动到JAX设备
        model.to('jax')
        
        # 对权重进行分片
        weights = shard_weights_llama(mesh, model.state_dict())
        
        # 将分片后的权重加载回模型
        model.load_state_dict(weights, assign=True, strict=False)
        
        # 对输入进行处理（复制到所有设备）
        input_ids = model_inputs.input_ids.to('jax').apply_jax_(
            jax.device_put,
            NamedSharding(mesh, P())
        )
        
        # 确保所有权重已分片完成
        tx.interop.call_jax(jax.block_until_ready, weights)
        
        # 6. 测试KV cache功能（可选）
        print("\n" + "=" * 60)
        print("6. 测试KV Cache功能")
        print("=" * 60)
        
        # 取消注释以运行这些测试
        run_twice_and_print_cache(model, input_ids)
        run_twice_and_print_cache_static(model, input_ids)
        
        # 7. 自回归解码（主要功能）
        print("\n" + "=" * 60)
        print("7. 自回归解码生成文本")
        print("=" * 60)
        
        # 首先使用动态缓存进行解码（作为基准）
        print("\n[方法1] 使用动态缓存（无JIT优化）")
        print("-" * 60)
        autoregressive_decode(model, input_ids, tokenizer, max_tokens=50)
        
        # 然后使用静态缓存和JIT优化进行解码
        print("\n[方法2] 使用静态缓存 + JIT优化")
        print("-" * 60)
        autoregressive_decode_static(model, input_ids, tokenizer, mesh, max_tokens=50)
    
    print("\n" + "=" * 60)
    print("解码完成！")
    print("=" * 60)
    print("\n关键特性：")
    print("- StaticCache: 预分配内存，避免动态分配开销")
    print("- JIT编译: 加速单token解码过程")
    print("- KV Cache分片: 在多设备上并行处理")
    print("- torch.func.functional_call: 无状态函数调用")


if __name__ == "__main__":
    main()