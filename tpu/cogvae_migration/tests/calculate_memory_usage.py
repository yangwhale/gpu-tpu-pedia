"""
计算 CogVideoX VAE decode 的理论显存使用
"""

def calculate_memory_usage(frames, height, width, dtype_bytes=2):
    """
    计算 VAE decode 的显存使用
    
    Args:
        frames: 输入帧数
        height: 视频高度
        width: 视频宽度
        dtype_bytes: 数据类型字节数 (bfloat16=2, float32=4)
    """
    print(f"\n{'='*70}")
    print(f"显存计算: {frames} 帧 @ {height}x{width}")
    print(f"数据类型: {'bfloat16' if dtype_bytes == 2 else 'float32'}")
    print(f"{'='*70}\n")
    
    # 常量
    latent_channels = 16
    temporal_compression = 4
    spatial_compression = 8
    
    # 计算 latent 尺寸
    latent_frames = max(1, frames // temporal_compression)
    latent_height = height // spatial_compression
    latent_width = width // spatial_compression
    
    print(f"1. 输入 Latent")
    print(f"   形状: (1, {latent_frames}, {latent_height}, {latent_width}, {latent_channels})")
    latent_size = 1 * latent_frames * latent_height * latent_width * latent_channels * dtype_bytes
    print(f"   大小: {latent_size / (1024**2):.2f} MB")
    
    # 计算输出尺寸（考虑 padding）
    # CogVideoX 的 Conv3d padding 会增加宽度
    output_width_padded = width + 16  # 根据之前的观察 720→736
    
    print(f"\n2. 输出 Video")
    print(f"   形状: (1, {frames}, {height}, {output_width_padded}, 3)")
    output_size = 1 * frames * height * output_width_padded * 3 * dtype_bytes
    print(f"   大小: {output_size / (1024**2):.2f} MB")
    
    # 计算中间激活值（最关键的部分）
    print(f"\n3. 中间激活值 (Decoder Feature Maps)")
    
    # Up blocks 的 channel 配置: [512, 256, 256, 128]
    up_block_channels = [512, 256, 256, 128]
    
    # 每个 up_block 有 3 个 resnet layers
    layers_per_block = 3
    
    max_activation_size = 0
    max_activation_info = ""
    total_activation_size = 0
    
    # 模拟 decoder 的处理流程
    current_t = latent_frames
    current_h = latent_height
    current_w = latent_width
    
    print(f"   Up Block 结构:")
    print(f"   {'Block':<15} {'Channels':<10} {'T':<5} {'H':<6} {'W':<6} {'Size (MB)':<12}")
    print(f"   {'-'*60}")
    
    for block_idx, channels in enumerate(up_block_channels):
        # 每个 block 的每个 layer
        for layer_idx in range(layers_per_block):
            # 计算当前层的激活值大小
            activation_size = 1 * current_t * current_h * current_w * channels * dtype_bytes
            activation_mb = activation_size / (1024**2)
            
            block_name = f"UpBlock{block_idx}[{layer_idx}]"
            print(f"   {block_name:<15} {channels:<10} {current_t:<5} {current_h:<6} {current_w:<6} {activation_mb:<12.2f}")
            
            total_activation_size += activation_size
            if activation_size > max_activation_size:
                max_activation_size = activation_size
                max_activation_info = f"{block_name}: {current_t}x{current_h}x{current_w}x{channels}"
        
        # Upsample 操作（在每个 block 后）
        if block_idx < len(up_block_channels) - 1:
            # 时间维度 upsample (如果需要)
            if current_t < frames:
                current_t = min(current_t * 2, frames)
            
            # 空间维度 upsample
            current_h *= 2
            current_w *= 2
    
    print(f"\n   最大激活: {max_activation_info}")
    print(f"   最大大小: {max_activation_size / (1024**2):.2f} MB")
    print(f"   累计总和: {total_activation_size / (1024**2):.2f} MB")
    
    # GroupNorm 临时内存
    # GroupNorm 需要转换到 channel-first 格式，可能需要额外副本
    print(f"\n4. GroupNorm 临时内存")
    groupnorm_temp = max_activation_size  # 最坏情况：需要完整副本
    print(f"   临时副本: {groupnorm_temp / (1024**2):.2f} MB")
    
    # Conv3d 操作的临时内存
    print(f"\n5. Conv3d 操作临时内存")
    conv_temp = max_activation_size * 1.5  # 估计需要 1.5x 用于卷积操作
    print(f"   临时缓存: {conv_temp / (1024**2):.2f} MB")
    
    # 总计
    print(f"\n{'='*70}")
    print(f"显存使用总结:")
    print(f"{'-'*70}")
    print(f"输入 Latent:        {latent_size / (1024**2):>10.2f} MB")
    print(f"输出 Video:         {output_size / (1024**2):>10.2f} MB")
    print(f"最大激活值:         {max_activation_size / (1024**2):>10.2f} MB")
    print(f"GroupNorm 临时:     {groupnorm_temp / (1024**2):>10.2f} MB")
    print(f"Conv3d 临时:        {conv_temp / (1024**2):>10.2f} MB")
    print(f"{'-'*70}")
    
    # 理论最小值（只计算峰值内存，不累加所有层）
    min_total = latent_size + output_size + max_activation_size + groupnorm_temp
    print(f"理论最小总计:       {min_total / (1024**2):>10.2f} MB ({min_total / (1024**3):.2f} GB)")
    
    # 实际使用（考虑多个中间层同时存在）
    # JAX/XLA 可能会保留多个中间层用于自动微分
    actual_total = latent_size + output_size + max_activation_size + groupnorm_temp + conv_temp
    print(f"实际峰值总计:       {actual_total / (1024**2):>10.2f} MB ({actual_total / (1024**3):.2f} GB)")
    
    # 在 replicated sharding 下，每个设备都需要完整副本
    print(f"\n在 Replicated Sharding 下 (8 TPU):")
    print(f"每个设备需要:       {actual_total / (1024**2):>10.2f} MB ({actual_total / (1024**3):.2f} GB)")
    print(f"总计 (8x):          {actual_total * 8 / (1024**2):>10.2f} MB ({actual_total * 8 / (1024**3):.2f} GB)")
    
    print(f"{'='*70}\n")
    
    return {
        'latent_mb': latent_size / (1024**2),
        'output_mb': output_size / (1024**2),
        'max_activation_mb': max_activation_size / (1024**2),
        'total_mb': actual_total / (1024**2),
        'total_gb': actual_total / (1024**3),
    }


if __name__ == "__main__":
    # 测试不同配置
    configs = [
        (8, 768, 1360, "8 帧 @ 768x1360 (你的测试)"),
        (4, 768, 1360, "4 帧 @ 768x1360 (更小)"),
        (16, 480, 720, "16 帧 @ 480x720 (之前的测试)"),
        (9, 480, 720, "9 帧 @ 480x720 (之前的测试)"),
    ]
    
    print("\n" + "="*70)
    print("CogVideoX VAE Decode 显存使用分析")
    print("="*70)
    
    results = []
    for frames, height, width, desc in configs:
        result = calculate_memory_usage(frames, height, width)
        result['config'] = desc
        results.append(result)
    
    # 总结对比
    print("\n" + "="*70)
    print("配置对比")
    print("="*70)
    print(f"{'配置':<30} {'总显存 (GB)':<15} {'峰值激活 (MB)':<20}")
    print("-"*70)
    for r in results:
        print(f"{r['config']:<30} {r['total_gb']:<15.2f} {r['max_activation_mb']:<20.2f}")
    print("="*70)