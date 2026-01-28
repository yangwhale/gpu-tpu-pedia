#!/usr/bin/env python3
"""
SGLang Installation Diagnostic Script

This script diagnoses common SGLang installation issues and provides fixes.
Run this when encountering import errors, library not found errors, or server startup failures.

Usage:
    python3 diagnose.py
    python3 diagnose.py --fix  # Attempt automatic fixes
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")

def print_ok(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_warn(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_fix(text):
    print(f"{Colors.BLUE}  Fix: {text}{Colors.RESET}")

def check_cuda():
    """Check CUDA installation."""
    print_header("CUDA Installation")

    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')

    # Check CUDA_HOME
    if os.path.exists(cuda_home):
        print_ok(f"CUDA_HOME exists: {cuda_home}")
    else:
        print_error(f"CUDA_HOME not found: {cuda_home}")
        print_fix("export CUDA_HOME=/usr/local/cuda")
        return False

    # Check nvcc
    nvcc_path = shutil.which('nvcc') or os.path.join(cuda_home, 'bin', 'nvcc')
    if os.path.exists(nvcc_path):
        result = subprocess.run([nvcc_path, '--version'], capture_output=True, text=True)
        version_line = [l for l in result.stdout.split('\n') if 'release' in l]
        if version_line:
            print_ok(f"nvcc found: {version_line[0].strip()}")
    else:
        print_warn(f"nvcc not in PATH")
        print_fix(f"export PATH={cuda_home}/bin:$PATH")

    return True

def check_pytorch():
    """Check PyTorch installation and CUDA support."""
    print_header("PyTorch Installation")

    try:
        import torch
        print_ok(f"PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            print_ok(f"CUDA available: True")
            print_ok(f"CUDA version: {torch.version.cuda}")
            print_ok(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print_ok(f"  GPU {i}: {props.name} ({props.total_memory // (1024**3)} GB)")
        else:
            print_error("CUDA not available in PyTorch")
            print_fix("pip install torch --extra-index-url https://download.pytorch.org/whl/cu129")
            return False

    except ImportError as e:
        print_error(f"PyTorch import failed: {e}")
        if 'libcudnn' in str(e):
            print_fix("pip install nvidia-cudnn-cu12==9.16.0.29 --force-reinstall --no-deps")
        elif 'libcusparseLt' in str(e):
            print_fix("pip install nvidia-cusparselt-cu12")
        return False

    return True

def check_deepep():
    """Check DeepEP installation - required for MoE models like DeepSeek."""
    print_header("DeepEP Installation")

    deepep_installed = False

    try:
        import deep_ep
        print_ok("DeepEP: OK (import deep_ep)")
        deepep_installed = True
    except ImportError:
        pass

    if not deepep_installed:
        try:
            # Some versions use different import name
            import deepep
            print_ok("DeepEP: OK (import deepep)")
            deepep_installed = True
        except ImportError:
            pass

    if not deepep_installed:
        # Check if DeepEP is in pip list
        try:
            result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
            if 'deep-ep' in result.stdout.lower() or 'deepep' in result.stdout.lower():
                print_ok("DeepEP: Found in pip packages")
                deepep_installed = True
        except:
            pass

    if not deepep_installed:
        # Check if DeepEP directory exists
        deepep_paths = [
            '/sgl-workspace/DeepEP',
            '/opt/deepep',
            os.path.expanduser('~/DeepEP'),
        ]
        for path in deepep_paths:
            if os.path.exists(path):
                print_ok(f"DeepEP: Found at {path}")
                deepep_installed = True
                break

    if not deepep_installed:
        print_warn("DeepEP: Not installed")
        print(f"{Colors.YELLOW}  DeepEP is required for MoE models (DeepSeek-V3, DeepSeek-R1){Colors.RESET}")
        print(f"{Colors.BLUE}  To install DeepEP, use the deepep-installer skill:{Colors.RESET}")
        print(f"{Colors.BLUE}    /deepep-installer{Colors.RESET}")
        print(f"{Colors.BLUE}  Or run the installation script:{Colors.RESET}")
        print(f"{Colors.BLUE}    bash /path/to/gpu-tpu-pedia/gpu/deepep/install.sh{Colors.RESET}")
        return False, "not_installed"

    # Check DeepEP dependencies
    gdrcopy_ok = os.path.exists('/opt/deepep/gdrcopy') or os.path.exists('/usr/src/gdrdrv-2.5.1')
    nvshmem_ok = os.path.exists('/opt/deepep/nvshmem') or 'nvshmem' in os.environ.get('LD_LIBRARY_PATH', '')

    if gdrcopy_ok:
        print_ok("GDRCopy: OK")
    else:
        print_warn("GDRCopy: Not found (may be optional)")

    if nvshmem_ok:
        print_ok("NVSHMEM: OK")
    else:
        print_warn("NVSHMEM: Not found (may be optional)")

    return True, "installed"


def check_sglang():
    """Check SGLang installation."""
    print_header("SGLang Installation")

    try:
        import sglang
        print_ok(f"SGLang version: {sglang.__version__}")
    except ImportError as e:
        print_error(f"SGLang import failed: {e}")
        print_fix("pip install -e 'python[blackwell]' --extra-index-url https://download.pytorch.org/whl/cu129")
        return False

    try:
        import sgl_kernel
        print_ok("sgl_kernel: OK")
    except ImportError as e:
        print_error(f"sgl_kernel import failed: {e}")
        print_fix("pip install sgl-kernel==0.3.21")
        return False

    try:
        import flashinfer
        print_ok("flashinfer: OK")
    except ImportError as e:
        print_warn(f"flashinfer import failed: {e}")

    return True

def check_nvidia_libs():
    """Check NVIDIA library installation and paths."""
    print_header("NVIDIA Libraries")

    required_libs = [
        ('libcudnn.so.9', 'nvidia-cudnn-cu12==9.16.0.29'),
        ('libcusparseLt.so.0', 'nvidia-cusparselt-cu12'),
        ('libnccl.so.2', 'nvidia-nccl-cu12==2.28.3'),
    ]

    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    print(f"LD_LIBRARY_PATH configured: {'Yes' if ld_path else 'No'}")

    # Find nvidia pip package lib directories
    nvidia_lib_dirs = []
    for base in ['/usr/local/lib/python3.12/dist-packages/nvidia',
                 os.path.expanduser('~/.local/lib/python3.12/site-packages/nvidia')]:
        if os.path.exists(base):
            for pkg in os.listdir(base):
                lib_dir = os.path.join(base, pkg, 'lib')
                if os.path.isdir(lib_dir):
                    nvidia_lib_dirs.append(lib_dir)

    print(f"Found {len(nvidia_lib_dirs)} nvidia lib directories")

    all_ok = True
    for lib_name, install_cmd in required_libs:
        found = False
        found_path = None

        # Search in nvidia lib dirs
        for lib_dir in nvidia_lib_dirs:
            lib_path = os.path.join(lib_dir, lib_name)
            if os.path.exists(lib_path):
                found = True
                found_path = lib_path
                break

        if found:
            print_ok(f"{lib_name}: Found at {found_path}")
            # Check if it's in LD_LIBRARY_PATH
            lib_dir = os.path.dirname(found_path)
            if lib_dir not in ld_path:
                print_warn(f"  {lib_dir} not in LD_LIBRARY_PATH")
        else:
            print_error(f"{lib_name}: Not found")
            print_fix(f"pip install {install_cmd} --force-reinstall --no-deps")
            all_ok = False

    if not all_ok or nvidia_lib_dirs:
        print("\n" + Colors.YELLOW + "To fix LD_LIBRARY_PATH issues, run:" + Colors.RESET)
        print("  source scripts/setup_env.sh")
        print("  # or add to your shell profile")

    return all_ok

def check_model_tp_compatibility(model_name=None):
    """Check model tensor parallelism compatibility."""
    print_header("Tensor Parallelism Compatibility")

    if model_name is None:
        print("No model specified. Common model head counts:")
        models = [
            ("Qwen2.5-7B", 28, [1, 2, 4, 7, 14]),
            ("Qwen2.5-72B", 64, [1, 2, 4, 8, 16, 32]),
            ("Llama-3-8B", 32, [1, 2, 4, 8, 16, 32]),
            ("Llama-3-70B", 64, [1, 2, 4, 8, 16, 32]),
            ("DeepSeek-R1", 128, [1, 2, 4, 8, 16, 32, 64]),
        ]
        for name, heads, valid_tp in models:
            print(f"  {name}: {heads} heads -> tp={valid_tp}")
    else:
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            heads = config.num_attention_heads

            # Calculate valid TP values
            valid_tp = [i for i in range(1, heads + 1) if heads % i == 0]

            print_ok(f"Model: {model_name}")
            print_ok(f"Attention heads: {heads}")
            print_ok(f"Valid TP values: {valid_tp[:10]}...")  # Show first 10

        except Exception as e:
            print_warn(f"Could not load model config: {e}")

def generate_fix_script():
    """Generate a script to fix common issues."""
    print_header("Fix Script")

    script = '''#!/bin/bash
# SGLang Fix Script - Generated by diagnose.py

set -e

echo "Installing required NVIDIA libraries..."
pip install nvidia-cudnn-cu12==9.16.0.29 --force-reinstall --no-deps
pip install nvidia-cusparselt-cu12 --force-reinstall --no-deps
pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps

echo "Setting up LD_LIBRARY_PATH..."
source scripts/setup_env.sh

echo "Verifying installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python3 -c "import sglang; print(f'SGLang: {sglang.__version__}')"

echo "Done! Environment is ready."
'''

    fix_script_path = Path(__file__).parent / 'fix_installation.sh'
    with open(fix_script_path, 'w') as f:
        f.write(script)
    os.chmod(fix_script_path, 0o755)

    print_ok(f"Fix script generated: {fix_script_path}")
    print(f"  Run: bash {fix_script_path}")

def main():
    print(f"{Colors.BOLD}SGLang Installation Diagnostics{Colors.RESET}")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")

    auto_fix = '--fix' in sys.argv
    check_deepep_flag = '--check-deepep' in sys.argv or '--deepep' in sys.argv

    # Run checks
    cuda_ok = check_cuda()
    pytorch_ok = check_pytorch() if cuda_ok else False
    sglang_ok = check_sglang() if pytorch_ok else False
    libs_ok = check_nvidia_libs()

    # DeepEP check (optional but important for MoE models)
    deepep_ok, deepep_status = check_deepep()

    check_model_tp_compatibility()

    # Summary
    print_header("Summary")

    all_ok = cuda_ok and pytorch_ok and sglang_ok and libs_ok

    if all_ok:
        print_ok("All checks passed! SGLang is ready to use.")
        if not deepep_ok:
            print_warn("Note: DeepEP not installed. Required for MoE models (DeepSeek-V3/R1)")
        print("\nTo start the server:")
        print("  source scripts/setup_env.sh")
        print("  python3 -m sglang.launch_server --model-path MODEL --tp TP_SIZE")
    else:
        print_error("Some checks failed. See above for fixes.")
        generate_fix_script()

        if auto_fix:
            print("\nAttempting automatic fixes...")
            subprocess.run(['bash', 'scripts/fix_installation.sh'])

    # Special handling for DeepEP
    if not deepep_ok and deepep_status == "not_installed":
        print_header("DeepEP Installation Required")
        print(f"""
{Colors.YELLOW}DeepEP is not installed but is required for MoE models like:{Colors.RESET}
  - DeepSeek-V3
  - DeepSeek-R1
  - Mixtral (with EP)

{Colors.BLUE}To install DeepEP, you can:{Colors.RESET}

  1. Use the deepep-installer skill (recommended):
     {Colors.GREEN}/deepep-installer{Colors.RESET}

  2. Run the installation script directly:
     {Colors.GREEN}bash /home/chrisya/gpu-tpu-pedia/gpu/deepep/install.sh{Colors.RESET}

  3. Follow the manual installation guide:
     {Colors.GREEN}See: gpu/deepep/README.md{Colors.RESET}

{Colors.YELLOW}After installing DeepEP, run this diagnostic again.{Colors.RESET}
""")

if __name__ == "__main__":
    main()
