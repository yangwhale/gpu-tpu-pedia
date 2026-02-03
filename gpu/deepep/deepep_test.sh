#!/usr/bin/env bash

# ============================================
# DeePP (深度专家并行) 性能测试脚本
# ============================================
# 功能：遍历多种参数组合，测试分布式专家系统的性能
# 
# 环境变量要求：
#   RANK: 当前节点的排名编号 (0, 1, 2, ...)
#   WORLD_SIZE: 集群中总节点数量
# ============================================

# --------------------------------------------
# 第一部分：定义测试参数
# --------------------------------------------

# 隐藏层维度数组 - 神经网络中间层的大小
# 不同的维度会影响模型的容量和计算量
n_hiddens=(4096 2048 7168 1024 6144 5120)

# Top-K 数组 - 每次选择激活的专家数量
# 例如：topk=4 表示从所有专家中选择4个最相关的专家
topk=(4 5 6 7 8 16 32)

# 每个节点的专家数量
# 节点越多，总专家数 = n_experts_per_node × WORLD_SIZE
n_experts_per_node=(32 64 128 256)

# 创建日志目录（如果不存在）
mkdir -p logs

# --------------------------------------------
# 第二部分：多层循环遍历所有参数组合
# --------------------------------------------

echo "======================================"
echo "开始 DeePP 性能测试"
echo "总节点数: ${WORLD_SIZE}"
echo "当前节点排名: ${RANK}"
echo "======================================"
echo ""

# 循环1：遍历不同的隐藏层维度
for nh in "${n_hiddens[@]}"; do
  
  # 循环2：遍历不同的 Top-K 值
  for tk in "${topk[@]}"; do
    
    # 循环3：遍历不同的每节点专家数量
    for ne in "${n_experts_per_node[@]}"; do
      
      # --------------------------------------------
      # 计算 topk_group 的上限
      # --------------------------------------------
      # topk_group 表示将 top-k 专家分成几组
      # 最大不能超过总节点数或 topk 值中的较小者
      if [ ${WORLD_SIZE} -gt ${tk} ]; then
        topk_group_limit=${tk}
      else
        topk_group_limit=${WORLD_SIZE}
      fi
      
      # 循环4：遍历不同的 topk_group 值（从1到上限）
      for ((topk_group=1; topk_group<=topk_group_limit; topk_group++)); do
        
        # --------------------------------------------
        # 打印当前测试配置
        # --------------------------------------------
        echo "----------------------------------------"
        echo "【测试配置】"
        echo "  隐藏层维度:        ${nh}"
        echo "  Top-K 专家数:      ${tk}"
        echo "  每节点专家数:      ${ne}"
        echo "  Top-K 分组数:      ${topk_group}"
        echo "  总专家数:          $((ne * WORLD_SIZE))"
        echo "----------------------------------------"
        
        # --------------------------------------------
        # 运行分布式训练
        # --------------------------------------------
        # torchrun 参数说明：
        #   --master-addr gpu-1:     主节点地址
        #   --master-port 45678:     主节点端口
        #   --node-rank ${RANK}:     当前节点排名
        #   --nproc_per_node=1:      每个节点使用1个进程
        #   --nnodes "${WORLD_SIZE}": 总节点数
        #
        # test_internode.py 参数说明：
        #   --hidden:              隐藏层维度
        #   --num-topk:            选择多少个专家
        #   --num-experts:         总专家数量
        #   --num-topk-groups:     将 top-k 专家分成几组
        
        torchrun \
          --master-addr gpu-1 \
          --master-port 45678 \
          --node-rank ${RANK} \
          --nproc_per_node=1 \
          --nnodes "${WORLD_SIZE}" \
          test_internode.py \
          --hidden "${nh}" \
          --num-topk "${tk}" \
          --num-experts $((ne * WORLD_SIZE)) \
          --num-topk-groups ${topk_group} | \
          tee ./logs/deepep_intranode_hidden_${nh}_topk_${tk}_expert_per_node_${ne}_topk_group_${topk_group}_nnodes_${WORLD_SIZE}.log
        
        echo "测试完成，日志已保存"
        echo ""
      done
    done
  done
done

echo "======================================"
echo "所有测试完成！"
echo "日志文件保存在 ./logs/ 目录"
echo "======================================"