#!/usr/bin/env python3
"""
Ray Cluster Validation Test Script
Usage: python3 test_cluster.py [HEAD_IP] [RAY_PORT]
"""

import ray
import time
import socket
import os
import sys

# Configuration
HEAD_IP = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("HEAD_IP", "10.8.0.79")
RAY_PORT = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("RAY_PORT", "6379"))


def print_separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


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

        gpu_count = resources.get("GPU", 0)
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
            alive = "Active" if node["Alive"] else "Offline"
            node_ip = node.get("NodeManagerAddress", "Unknown")
            resources = node.get("Resources", {})
            gpus = resources.get("GPU", 0)
            cpus = resources.get("CPU", 0)

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
    return {"hostname": hostname, "ip": ip, "pid": pid}


@ray.remote(num_gpus=1)
def get_gpu_info():
    """Remote function: get GPU info"""
    import subprocess

    hostname = socket.gethostname()
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpu_info = result.stdout.strip()
    except Exception as e:
        gpu_info = f"Error: {e}"
    return {"hostname": hostname, "gpu_info": gpu_info}


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
        gpu_count = int(resources.get("GPU", 0))

        if gpu_count == 0:
            print("⚠ No GPUs detected in cluster")
            return True

        print(f"\nExecuting tasks on {min(gpu_count, 16)} GPUs...")
        num_tasks = min(gpu_count, 16)
        futures = [get_gpu_info.remote() for _ in range(num_tasks)]
        results = ray.get(futures, timeout=60)

        by_node = {}
        for r in results:
            node = r["hostname"]
            if node not in by_node:
                by_node[node] = []
            by_node[node].append(r["gpu_info"])

        for node, gpus in by_node.items():
            print(f"\n{node} GPUs:")
            for gpu in set(gpus):
                for line in gpu.split("\n"):
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
    print("\n" + "=" * 60)
    print("         Ray Cluster Validation")
    print("=" * 60)
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
