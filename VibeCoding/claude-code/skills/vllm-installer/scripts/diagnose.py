#!/usr/bin/env python3
"""
vLLM Installation Diagnostic Script

This script checks the installation status of vLLM and its dependencies,
including LSSD mount status and DeepEP for MoE models.

Usage:
    python3 diagnose.py
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd: str, capture: bool = True) -> tuple[int, str]:
    """Run a shell command and return exit code and output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=capture, text=True, timeout=30
        )
        return result.returncode, result.stdout.strip() if capture else ""
    except subprocess.TimeoutExpired:
        return 1, "Command timed out"
    except Exception as e:
        return 1, str(e)


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_ok(msg: str):
    print(f"  \033[32m✓\033[0m {msg}")


def print_warn(msg: str):
    print(f"  \033[33m⚠\033[0m {msg}")


def print_error(msg: str):
    print(f"  \033[31m✗\033[0m {msg}")


def print_info(msg: str):
    print(f"    {msg}")


def check_lssd():
    """Check if LSSD is mounted."""
    print_header("LSSD Status")

    # Check if /lssd is mounted
    code, _ = run_command("mountpoint -q /lssd")
    if code == 0:
        # Get mount info
        _, df_output = run_command("df -h /lssd | tail -1")
        parts = df_output.split()
        if len(parts) >= 4:
            size, used, avail, use_pct = parts[1:5]
            print_ok(f"LSSD mounted: {size} total, {avail} available ({use_pct} used)")
        else:
            print_ok("LSSD mounted")

        # Check HF_HOME
        hf_home = os.environ.get("HF_HOME", "")
        if hf_home == "/lssd/huggingface":
            print_ok(f"HF_HOME set to {hf_home}")
        elif os.path.exists("/lssd/huggingface"):
            print_warn("HF_HOME not set to /lssd/huggingface")
            print_info("Run: export HF_HOME=/lssd/huggingface")
        return True
    else:
        # Check if NVMe SSDs exist
        _, nvme_count = run_command(
            "ls /dev/disk/by-id/ 2>/dev/null | grep -c 'google-local-nvme-ssd' || echo 0"
        )
        nvme_count = int(nvme_count) if nvme_count.isdigit() else 0

        if nvme_count > 0:
            print_warn(f"LSSD not mounted ({nvme_count} NVMe SSDs detected)")
            print_info("To mount LSSD, use the lssd-mounter skill:")
            print_info("  /lssd-mounter")
        else:
            print_info("No Local SSD detected (VM may not have LSSD attached)")
        return False


def check_cuda():
    """Check CUDA installation."""
    print_header("CUDA Environment")

    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")

    # Check nvcc
    code, version = run_command(f"{cuda_home}/bin/nvcc --version 2>/dev/null | grep 'release'")
    if code == 0:
        print_ok(f"CUDA: {version.split('release')[-1].strip().split(',')[0] if 'release' in version else 'installed'}")
    else:
        print_error("CUDA not found or CUDA_HOME not set")
        print_info(f"CUDA_HOME: {cuda_home}")
        return False

    # Check nvidia-smi
    code, gpu_info = run_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1")
    if code == 0:
        print_ok(f"GPU: {gpu_info}")
        _, gpu_count = run_command("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l")
        print_ok(f"GPU count: {gpu_count}")
    else:
        print_error("nvidia-smi failed")
        return False

    return True


def check_pytorch():
    """Check PyTorch installation."""
    print_header("PyTorch")

    try:
        import torch
        print_ok(f"PyTorch: {torch.__version__}")

        if torch.cuda.is_available():
            print_ok(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
        else:
            print_error("CUDA not available in PyTorch")
            return False
        return True
    except ImportError as e:
        print_error(f"PyTorch not installed: {e}")
        return False


def check_vllm():
    """Check vLLM installation."""
    print_header("vLLM")

    try:
        import vllm
        print_ok(f"vLLM: {vllm.__version__}")
        return True
    except ImportError as e:
        print_error(f"vLLM not installed: {e}")
        print_info("Install with: pip install vllm==0.14.1")
        return False


def check_flashinfer():
    """Check FlashInfer installation."""
    print_header("FlashInfer")

    # Check flashinfer-python
    code_py, version_py = run_command("pip show flashinfer-python 2>/dev/null | grep 'Version:'")
    if code_py == 0:
        ver_py = version_py.split(":")[-1].strip()
        print_ok(f"flashinfer-python: {ver_py}")
    else:
        print_error("flashinfer-python not installed")
        print_info("Install with: pip install flashinfer-python==0.5.3")
        return False

    # Check flashinfer-cubin
    code_cubin, version_cubin = run_command("pip show flashinfer-cubin 2>/dev/null | grep 'Version:'")
    if code_cubin == 0:
        ver_cubin = version_cubin.split(":")[-1].strip()
        print_ok(f"flashinfer-cubin: {ver_cubin}")
    else:
        print_error("flashinfer-cubin not installed")
        print_info("Install with: pip install flashinfer-cubin==0.5.3")
        return False

    # Check version match
    if ver_py != ver_cubin:
        print_error(f"Version mismatch! python={ver_py}, cubin={ver_cubin}")
        print_info("Both versions MUST match. Fix with:")
        print_info(f"  pip install flashinfer-python=={ver_py} flashinfer-cubin=={ver_py} --force-reinstall")
        return False

    # Test import
    try:
        import flashinfer
        print_ok("FlashInfer import: OK")
        return True
    except ImportError as e:
        print_error(f"FlashInfer import failed: {e}")
        print_info("Check LD_LIBRARY_PATH configuration")
        return False


def check_nvidia_libs():
    """Check NVIDIA library installations."""
    print_header("NVIDIA Libraries")

    libs = [
        ("nvidia-nccl-cu12", "2.28.3"),
        ("nvidia-cudnn-cu12", "9.16.0.29"),
        ("nvidia-cusparselt-cu12", None),
    ]

    all_ok = True
    for lib, expected_version in libs:
        code, version = run_command(f"pip show {lib} 2>/dev/null | grep 'Version:'")
        if code == 0:
            ver = version.split(":")[-1].strip()
            if expected_version and ver != expected_version:
                print_warn(f"{lib}: {ver} (expected {expected_version})")
                all_ok = False
            else:
                print_ok(f"{lib}: {ver}")
        else:
            print_error(f"{lib}: not installed")
            if expected_version:
                print_info(f"Install with: pip install {lib}=={expected_version} --force-reinstall --no-deps")
            else:
                print_info(f"Install with: pip install {lib} --force-reinstall --no-deps")
            all_ok = False

    return all_ok


def check_ld_library_path():
    """Check LD_LIBRARY_PATH configuration."""
    print_header("LD_LIBRARY_PATH")

    ld_path = os.environ.get("LD_LIBRARY_PATH", "")

    if not ld_path:
        print_warn("LD_LIBRARY_PATH is not set")
        print_info("Run: source /vllm-workspace/vllm-env.sh")
        return False

    # Check for nvidia paths
    nvidia_paths = [p for p in ld_path.split(":") if "nvidia" in p.lower()]
    if nvidia_paths:
        print_ok(f"NVIDIA lib paths: {len(nvidia_paths)} found")
    else:
        print_warn("No NVIDIA lib paths in LD_LIBRARY_PATH")
        print_info("This may cause library loading errors")
        return False

    # Check for CUDA path
    if "/usr/local/cuda" in ld_path:
        print_ok("CUDA lib path included")
    else:
        print_warn("CUDA lib path not in LD_LIBRARY_PATH")

    return True


def check_deepep():
    """Check DeepEP installation (for MoE models)."""
    print_header("DeepEP (for MoE models)")

    try:
        import deep_ep
        print_ok("DeepEP installed (deep_ep module)")
        return True
    except ImportError:
        pass

    try:
        import deepep
        print_ok("DeepEP installed (deepep module)")
        return True
    except ImportError:
        pass

    # Check if installation directories exist
    deepep_dir = Path("/opt/deepep")
    if deepep_dir.exists():
        print_warn("DeepEP directory exists but module not importable")
        print_info("Check PYTHONPATH and LD_LIBRARY_PATH")
    else:
        print_info("DeepEP not installed")
        print_info("DeepEP is required for MoE models (DeepSeek-V3, DeepSeek-R1)")
        print_info("To install DeepEP, use the deepep-installer skill:")
        print_info("  /deepep-installer")

    return False


def main():
    print("\n" + "="*60)
    print(" vLLM Installation Diagnostic")
    print("="*60)

    results = {}

    # Run all checks
    results["lssd"] = check_lssd()
    results["cuda"] = check_cuda()
    results["pytorch"] = check_pytorch()
    results["vllm"] = check_vllm()
    results["flashinfer"] = check_flashinfer()
    results["nvidia_libs"] = check_nvidia_libs()
    results["ld_library_path"] = check_ld_library_path()
    results["deepep"] = check_deepep()

    # Summary
    print_header("Summary")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\n  Checks passed: {passed}/{total}")

    if all(results.values()):
        print("\n  \033[32m✓ All checks passed! vLLM should be ready to use.\033[0m")
        print("\n  Start server with:")
        print("    vllm serve Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 4 --port 8000")
    else:
        print("\n  \033[33m⚠ Some checks failed. Review the issues above.\033[0m")

        # Specific recommendations
        if not results["lssd"]:
            print("\n  → For high-speed model caching, mount LSSD:")
            print("      /lssd-mounter")

        if not results["deepep"]:
            print("\n  → For MoE models (DeepSeek-V3/R1), install DeepEP:")
            print("      /deepep-installer")

        if not results["ld_library_path"]:
            print("\n  → Fix LD_LIBRARY_PATH by running:")
            print("      source /vllm-workspace/vllm-env.sh")

    print()
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
