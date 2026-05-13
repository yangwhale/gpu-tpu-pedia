#!/bin/bash
set -e
hostfile=$1
if ! [ -s "$hostfile" ]; then
  echo "hostfile $hostfile doesn't exits"
  exit 1
fi
full=$(grep -v '#' $hostfile | awk 'NF==1' | wc -l)
from=1
to=$full
if [ -n "$2" ]; then
  from=$2
  if [ -n "$3" ]; then
    to=$3
  fi
fi
nnode=$((to + 1 - from))
hosts=$(grep -v '#' $hostfile | awk 'NF==1' | tr -d '\r' | awk "NR>=$from&&NR<=$to"'{print $1}')
master=$(echo $hosts | xargs -n1 | awk 'NR==1')
ts=$(date +%Y%m%d%H%M%S)
mode=${mode:-inter}
script=${script:-test_internode.py}
if [ $mode == "ll" ]; then
  script=test_low_latency.py
fi
if [ $nnode -eq 1 ] && [ $mode = "inter" ]; then
  mode=intra
  script=test_intranode.py
fi
tc_env=
if [ -n "$tc" ]; then
  tc_env="NVSHMEM_IB_TRAFFIC_CLASS=$tc"
fi
env="NVSHMEM_IB_GID_INDEX=3 DEEP_EP_DEVICE_TO_HCA_MAPPING=0:mlx5_0:1,1:mlx5_1:1,2:mlx5_2:1,3:mlx5_3:1,4:mlx5_4:1,5:mlx5_5:1,6:mlx5_6:1,7:mlx5_7:1 NCCL_IB_HCA==mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1 NCCL_IB_GID_INDEX=3 NVSHMEM_DEBUG=INFO NVSHMEM_DEBUG_SUBSYS=INIT,P2P,TRANSPORT,TOPO,UTIL NVSHMEM_IB_GID_INDEX=3 NVSHMEM_ENABLE_NIC_PE_MAPPING=1 NVSHMEM_DISABLE_CUDA_VMM=1"
node_rank=0
for h in $hosts; do
  out=/tmp/deepep-$mode-$hostfile-from$from-to$to-$ts-node$node_rank.log
  ssh -p ${SSH_PORT:-1257} -o StrictHostKeyChecking=no $h env $tc_env RANK=$node_rank WORLD_SIZE=$nnode MASTER_ADDR=$master MASTER_PORT=${MASTER_PORT:-29500} ts=$ts $env python3 /workspace/DeepEP/tests/$script $args 2>&1 | tee $out &
  if [ $node_rank -eq 0 ]; then
    sleep 3
  fi
  node_rank=$((node_rank + 1))
done
wait
echo /tmp/deepep-$mode-$hostfile-from$from-to$to-$ts-node0.log
if [ $mode == "inter" ]; then
  grep Best /tmp/deepep-$mode-$hostfile-from$from-to$to-$ts-node0.log
  # | sed -re 's/.*Best ([^:]+):.*RDMA chunk [0-9]+: (\S+).*/\1\t\2/' | awk -F'\t' '{s[$1]+=$2;c[$1]++}END{for(i in s)print i"\t"s[i]"\t"c[i]"\t"s[i]/c[i]}'
else
  grep -F 'us | Combine bandwidth: ' /tmp/deepep-$mode-$hostfile-from$from-to$to-$ts-node0.log | grep -F '[rank 0]'
  # | tr '|' '\n' | sed -re 's/.* (\S+) bandwidth: (\S+) .*=(\S+) us/\1\t\2\t\3/'
fi