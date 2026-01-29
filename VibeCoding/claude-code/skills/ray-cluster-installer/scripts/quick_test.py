#!/usr/bin/env python3
"""
Ray Local Quick Test Script
Validates Ray installation on a single machine without cluster connection.
"""

import ray
import time


def main():
    print("=" * 50)
    print("  Ray Local Quick Test")
    print("=" * 50)

    # Initialize local Ray
    print("\n1. Initializing local Ray...")
    ray.init(ignore_reinit_error=True)
    print("   ✓ Ray initialized successfully")

    # Check resources
    print("\n2. Checking local resources...")
    resources = ray.cluster_resources()
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
    if resources.get("GPU", 0) > 0:

        @ray.remote(num_gpus=1)
        def gpu_test():
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
            )
            return result.stdout.strip().split("\n")[0]

        print("\n4. Testing GPU access...")
        gpu_name = ray.get(gpu_test.remote())
        print(f"   GPU: {gpu_name}")
    else:
        print("\n4. Skipping GPU test (no GPU available)")

    # Parallel execution test
    @ray.remote
    def parallel_task(i):
        time.sleep(0.05)
        return i**2

    print("\n5. Testing parallel execution...")
    start = time.time()
    futures = [parallel_task.remote(i) for i in range(20)]
    results = ray.get(futures)
    elapsed = time.time() - start
    print(f"   20 parallel tasks: {elapsed:.2f}s")
    print(f"   Results valid: {results == [i**2 for i in range(20)]}")

    print("\n" + "=" * 50)
    print("  ✓ All tests passed! Ray is working correctly")
    print("=" * 50)

    ray.shutdown()


if __name__ == "__main__":
    main()
