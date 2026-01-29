---
name: ray-cluster-installer
description: Install, configure, and test Ray distributed clusters on NVIDIA GPU systems. Covers multi-node setup, GPU detection, cluster validation, and SGLang PD disaggregation deployment.
version: 1.0.0
source: local-skill-creation
author: chrisya
tags:
  - ray
  - distributed-computing
  - gpu
  - sglang
  - inference
---

# Ray Cluster Installer

This skill should be used when users need to install, configure, debug, or test Ray distributed clusters on NVIDIA GPU systems (especially B200/H100/A100). It covers installation verification, multi-node cluster setup, GPU resource detection, cluster validation testing, and integration with SGLang PD (Prefill-Decode) disaggregation.

## When to Use This Skill

- User wants to set up a Ray cluster across multiple machines
- User needs to verify Ray installation and GPU detection
- User wants to test distributed task execution across nodes
- User is deploying SGLang with PD disaggregation using Ray
- User needs to troubleshoot Ray cluster connectivity issues

## Prerequisites Check

Before starting, verify the environment:

```bash
# Check Python version (requires 3.8+)
python3 --version

# Check if Ray is installed
pip3 show ray

# Check GPU availability
nvidia-smi --query-gpu=name,memory.total --format=csv

# Check network interface for cluster communication
ip addr show | grep -E "inet.*enp|inet.*eth"

# Get current machine IP
hostname -I | awk '{print $1}'
```

## Installation (if Ray not installed)

```bash
# Install Ray with default dependencies
pip3 install -U ray[default]

# For GPU support, ensure CUDA is available
pip3 install ray[default] torch  # PyTorch for GPU testing
```

## Cluster Architecture

Ray uses a Head-Worker architecture:

| Role | Description |
|------|-------------|
| **Head Node** | Runs GCS (Global Control Store), manages cluster state, schedules tasks |
| **Worker Node** | Connects to Head, executes distributed tasks |

## Step 1: Start Head Node

On the designated Head machine (e.g., the prefill node for SGLang):

```bash
#!/bin/bash
# Ray Head Node startup script

HEAD_IP="${HEAD_IP:-$(hostname -I | awk '{print $1}')}"
RAY_PORT="${RAY_PORT:-6379}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_CPUS="${NUM_CPUS:-$(nproc)}"

# Stop existing Ray processes
ray stop --force 2>/dev/null || true
sleep 2

# Start Head Node
ray start --head \
    --port=$RAY_PORT \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=$DASHBOARD_PORT \
    --node-ip-address=$HEAD_IP \
    --num-cpus=$NUM_CPUS \
    --num-gpus=$NUM_GPUS

echo "Head Node started at $HEAD_IP:$RAY_PORT"
echo "Dashboard: http://$HEAD_IP:$DASHBOARD_PORT"
```

## Step 2: Start Worker Nodes

On each Worker machine:

```bash
#!/bin/bash
# Ray Worker Node startup script

HEAD_IP="${HEAD_IP:?HEAD_IP environment variable required}"
RAY_PORT="${RAY_PORT:-6379}"
WORKER_IP="${WORKER_IP:-$(hostname -I | awk '{print $1}')}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_CPUS="${NUM_CPUS:-$(nproc)}"

# Stop existing Ray processes
ray stop --force 2>/dev/null || true
sleep 2

# Start Worker Node
ray start \
    --address="$HEAD_IP:$RAY_PORT" \
    --node-ip-address=$WORKER_IP \
    --num-cpus=$NUM_CPUS \
    --num-gpus=$NUM_GPUS

echo "Worker Node connected to $HEAD_IP:$RAY_PORT"
```

## Step 3: Validate Cluster

### Quick Local Test (Single Machine)

```python
#!/usr/bin/env python3
"""Quick local Ray validation test"""
import ray
import time

def quick_local_test():
    print("=" * 50)
    print("  Ray Local Quick Test")
    print("=" * 50)

    # Initialize local Ray
    ray.init(ignore_reinit_error=True)
    print("\n1. Ray initialized successfully")

    # Check resources
    resources = ray.cluster_resources()
    print(f"\n2. Resources:")
    print(f"   CPU: {resources.get('CPU', 0)}")
    print(f"   GPU: {resources.get('GPU', 0)}")
    print(f"   Memory: {resources.get('memory', 0) / 1e9:.1f} GB")

    # Test remote execution
    @ray.remote
    def hello_ray(x):
        return f"Hello from Ray! Result: {x * 2}"

    print("\n3. Testing remote execution...")
    result = ray.get(hello_ray.remote(21))
    print(f"   {result}")

    # GPU test if available
    if resources.get('GPU', 0) > 0:
        @ray.remote(num_gpus=1)
        def gpu_test():
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, text=True
            )
            return result.stdout.strip().split('\n')[0]

        print("\n4. Testing GPU access...")
        gpu_name = ray.get(gpu_test.remote())
        print(f"   GPU: {gpu_name}")

    # Parallel execution test
    @ray.remote
    def parallel_task(i):
        time.sleep(0.05)
        return i ** 2

    print("\n5. Testing parallel execution...")
    start = time.time()
    futures = [parallel_task.remote(i) for i in range(20)]
    results = ray.get(futures)
    elapsed = time.time() - start
    print(f"   20 parallel tasks: {elapsed:.2f}s")
    print(f"   Results valid: {results == [i**2 for i in range(20)]}")

    print("\n" + "=" * 50)
    print("  All tests passed!")
    print("=" * 50)

    ray.shutdown()

if __name__ == "__main__":
    quick_local_test()
```

### Full Cluster Test (Multi-Node)

```python
#!/usr/bin/env python3
"""
Ray Cluster Validation Test
Tests multi-node cluster connectivity, resource detection, and distributed execution
"""
import ray
import time
import socket
import os
import sys

# Configuration - update these for your cluster
HEAD_IP = "10.8.0.79"  # Change to your head node IP
RAY_PORT = 6379

def print_separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def test_connection():
    """Test Ray cluster connection"""
    print_separator("Test 1: Cluster Connection")
    try:
        ray.init(address=f"ray://{HEAD_IP}:{RAY_PORT}", ignore_reinit_error=True)
        print(f"✓ Connected to Ray cluster: {HEAD_IP}:{RAY_PORT}")
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

def test_cluster_resources():
    """Test cluster resources"""
    print_separator("Test 2: Cluster Resources")
    try:
        resources = ray.cluster_resources()
        available = ray.available_resources()

        print("\nTotal cluster resources:")
        for key, value in sorted(resources.items()):
            print(f"  {key}: {value}")

        gpu_count = resources.get('GPU', 0)
        print(f"\n✓ Total GPUs in cluster: {gpu_count}")
        return True
    except Exception as e:
        print(f"✗ Resource detection failed: {e}")
        return False

def test_nodes():
    """Test node information"""
    print_separator("Test 3: Node Information")
    try:
        nodes = ray.nodes()
        print(f"\nCluster has {len(nodes)} nodes:\n")

        for i, node in enumerate(nodes):
            alive = "Active" if node['Alive'] else "Offline"
            node_ip = node.get('NodeManagerAddress', 'Unknown')
            resources = node.get('Resources', {})
            gpus = resources.get('GPU', 0)
            cpus = resources.get('CPU', 0)

            print(f"  Node {i+1}:")
            print(f"    - IP: {node_ip}")
            print(f"    - Status: {alive}")
            print(f"    - CPU: {cpus}")
            print(f"    - GPU: {gpus}")
            print()

        return True
    except Exception as e:
        print(f"✗ Node detection failed: {e}")
        return False

@ray.remote
def get_node_info():
    """Remote function: get execution node info"""
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    pid = os.getpid()
    return {'hostname': hostname, 'ip': ip, 'pid': pid}

@ray.remote(num_gpus=1)
def get_gpu_info():
    """Remote function: get GPU info"""
    import subprocess
    hostname = socket.gethostname()
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        gpu_info = result.stdout.strip()
    except Exception as e:
        gpu_info = f"Error: {e}"
    return {'hostname': hostname, 'gpu_info': gpu_info}

def test_remote_execution():
    """Test cross-node task execution"""
    print_separator("Test 4: Cross-Node Execution")
    try:
        print("\nSubmitting 8 tasks to cluster...")
        futures = [get_node_info.remote() for _ in range(8)]
        results = ray.get(futures)

        node_tasks = {}
        for r in results:
            key = f"{r['hostname']} ({r['ip']})"
            node_tasks[key] = node_tasks.get(key, 0) + 1

        print("\nTask distribution:")
        for node, count in node_tasks.items():
            print(f"  {node}: {count} tasks")

        if len(node_tasks) > 1:
            print("\n✓ Tasks distributed across multiple nodes")
        else:
            print("\n⚠ All tasks on single node (only one node may be online)")

        return True
    except Exception as e:
        print(f"✗ Remote execution failed: {e}")
        return False

def test_gpu_access():
    """Test GPU access across cluster"""
    print_separator("Test 5: GPU Access")
    try:
        resources = ray.cluster_resources()
        gpu_count = int(resources.get('GPU', 0))

        if gpu_count == 0:
            print("⚠ No GPUs detected in cluster")
            return True

        print(f"\nExecuting tasks on {min(gpu_count, 16)} GPUs...")
        num_tasks = min(gpu_count, 16)
        futures = [get_gpu_info.remote() for _ in range(num_tasks)]
        results = ray.get(futures, timeout=60)

        by_node = {}
        for r in results:
            node = r['hostname']
            if node not in by_node:
                by_node[node] = []
            by_node[node].append(r['gpu_info'])

        for node, gpus in by_node.items():
            print(f"\n{node} GPUs:")
            for gpu in set(gpus):
                for line in gpu.split('\n'):
                    if line.strip():
                        print(f"    {line.strip()}")

        print(f"\n✓ GPU access test completed")
        return True
    except Exception as e:
        print(f"✗ GPU access failed: {e}")
        return False

@ray.remote
def compute_task(x):
    """Simple compute task"""
    time.sleep(0.1)
    return x * x

def test_performance():
    """Performance test"""
    print_separator("Test 6: Parallel Performance")
    try:
        num_tasks = 100
        print(f"\nExecuting {num_tasks} parallel compute tasks...")

        start_time = time.time()
        futures = [compute_task.remote(i) for i in range(num_tasks)]
        results = ray.get(futures)
        end_time = time.time()

        elapsed = end_time - start_time
        throughput = num_tasks / elapsed

        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.1f} tasks/sec")
        print(f"  Results valid: {sum(results) == sum(i*i for i in range(num_tasks))}")

        print("\n✓ Performance test completed")
        return True
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("         Ray Cluster Validation")
    print("="*60)
    print(f"\nHead Node: {HEAD_IP}:{RAY_PORT}")
    print(f"Local IP: {socket.gethostbyname(socket.gethostname())}")

    tests = [
        ("Connection", test_connection),
        ("Resources", test_cluster_resources),
        ("Nodes", test_nodes),
        ("Remote Execution", test_remote_execution),
        ("GPU Access", test_gpu_access),
        ("Performance", test_performance),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"✗ {name} exception: {e}")
            results[name] = False

        if name == "Connection" and not results[name]:
            print("\nConnection failed, skipping remaining tests")
            break

    print_separator("Test Summary")
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    ray.shutdown()
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
```

## SGLang PD Disaggregation Integration

When deploying SGLang with Prefill-Decode disaggregation:

### Environment Variables

```bash
# Network configuration
export SGLANG_LOCAL_IP_NIC=enp0s19      # Network interface for SGLANG
export GLOO_SOCKET_IFNAME=enp0s19       # Gloo communication interface
export NCCL_SOCKET_IFNAME=enp0s19       # NCCL communication interface
export NCCL_MNNVL_ENABLE=1              # Enable NVLink if available
export NCCL_CUMEM_ENABLE=1              # Enable CUDA unified memory

# SGLang disaggregation settings
export SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000
```

### Prefill Node Configuration

```bash
# Prefill node (typically runs on Head)
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --disaggregation-mode prefill \
    --dist-init-addr ${HEAD_IP}:5757 \
    --nnodes 1 \
    --node-rank 0 \
    --tp-size 8 \
    --dp-size 8 \
    --enable-dp-attention \
    --host 0.0.0.0 \
    --deepep-mode normal \
    --disaggregation-transfer-backend nixl
```

### Decode Node Configuration

```bash
# Decode node (typically runs on Workers)
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --disaggregation-mode decode \
    --dist-init-addr ${HEAD_IP}:5757 \
    --nnodes ${NUM_DECODE_NODES} \
    --node-rank ${RANK} \
    --tp-size ${TP_SIZE} \
    --dp-size ${DP_SIZE} \
    --enable-dp-attention \
    --host 0.0.0.0 \
    --deepep-mode low_latency \
    --disaggregation-transfer-backend nixl
```

### Load Balancer

```bash
python3 -m sglang.srt.disaggregation.mini_lb \
    --prefill "http://${PREFILL_IP}:30000" \
    --decode "http://${DECODE_IP}:30000"
```

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| **Connection timeout** | Check firewall: ports 6379, 8265 must be open |
| **GPU not detected** | Verify CUDA and nvidia-driver installation |
| **Node won't join** | Verify HEAD_IP is correct, test with `ping` |
| **Tasks not distributed** | Check all nodes show as "Alive" in `ray.nodes()` |

### Diagnostic Commands

```bash
# Check Ray status
ray status

# View Ray logs
tail -f /tmp/ray/session_latest/logs/raylet.out

# Test network connectivity
ping ${HEAD_IP}

# Check if ports are open
nc -zv ${HEAD_IP} 6379
nc -zv ${HEAD_IP} 8265

# Stop Ray completely
ray stop --force
```

## Quick Reference

```bash
# Start head (on head machine)
ray start --head --port=6379 --dashboard-host=0.0.0.0 --num-gpus=8

# Join cluster (on worker machines)
ray start --address="${HEAD_IP}:6379" --num-gpus=8

# Check cluster status
ray status

# Stop Ray
ray stop --force

# View dashboard
# Open http://${HEAD_IP}:8265 in browser
```
