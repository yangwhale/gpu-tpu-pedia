#!/usr/bin/env python3
"""
ComfyUI on TPU - 一键安装脚本

使用方法:
    # 0. 首先克隆 gpu-tpu-pedia 仓库
    # git clone https://github.com/yangwhale/gpu-tpu-pedia.git
    # cd gpu-tpu-pedia/tpu/ComfyUI
    
    # 1. 运行安装脚本
    python3 setup.py
    
    # 2. 安装完成后，重新登录或执行:
    source ~/.bashrc
    
    # 3. 启动 ComfyUI
    cd ~/ComfyUI && python main.py --cpu --listen 0.0.0.0

环境要求:
    - Google Cloud TPU VM (v6e-8)
    - Ubuntu 22.04
    - 需要 sudo 权限

作者: Chris Yang
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_cmd(cmd: str, check: bool = True, shell: bool = True, cwd: str = None) -> subprocess.CompletedProcess:
    """执行 shell 命令"""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=shell, cwd=cwd, capture_output=False)
    if check and result.returncode != 0:
        print(f"[ERROR] 命令执行失败: {cmd}")
        sys.exit(1)
    return result


def run_cmd_output(cmd: str) -> str:
    """执行命令并返回输出"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()


def check_python_version():
    """检查 Python 版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"[ERROR] Python 版本过低: {version.major}.{version.minor}")
        print("需要 Python 3.10+，建议 3.12")
        return False
    return True


def install_python312():
    """安装 Python 3.12"""
    print("\n" + "="*60)
    print("步骤 1: 安装 Python 3.12")
    print("="*60)
    
    # 检查是否已安装
    result = run_cmd_output("which python3.12")
    if result:
        print(f"✓ Python 3.12 已安装: {result}")
        return
    
    # 停止 unattended-upgrades（避免 apt lock）
    run_cmd("sudo systemctl stop unattended-upgrades", check=False)
    
    # 添加 deadsnakes PPA
    run_cmd("sudo add-apt-repository ppa:deadsnakes/ppa -y")
    
    # 安装 Python 3.12
    run_cmd("sudo apt-get update")
    run_cmd("sudo apt-get install -y python3.12 python3.12-venv python3.12-dev")
    
    # 初始化 pip（Python 3.12 移除了 distutils）
    run_cmd("python3.12 -m ensurepip --upgrade")
    
    # 设置为默认 python
    run_cmd("sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1")
    run_cmd("sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1")
    
    print("✓ Python 3.12 安装完成")


def configure_pip():
    """配置 pip 允许系统级安装"""
    print("\n" + "="*60)
    print("步骤 2: 配置 pip")
    print("="*60)
    
    pip_conf_dir = Path.home() / ".config" / "pip"
    pip_conf_file = pip_conf_dir / "pip.conf"
    
    pip_conf_dir.mkdir(parents=True, exist_ok=True)
    
    pip_conf_content = """[global]
break-system-packages = true
"""
    
    pip_conf_file.write_text(pip_conf_content)
    print(f"✓ pip 配置已写入: {pip_conf_file}")


def install_comfyui():
    """安装 ComfyUI"""
    print("\n" + "="*60)
    print("步骤 3: 安装 ComfyUI")
    print("="*60)
    
    comfyui_path = Path.home() / "ComfyUI"
    
    if comfyui_path.exists():
        print(f"✓ ComfyUI 已存在: {comfyui_path}")
    else:
        run_cmd("git clone https://github.com/comfyanonymous/ComfyUI.git", cwd=str(Path.home()))
    
    # 安装依赖
    run_cmd("pip install -r requirements.txt", cwd=str(comfyui_path))
    
    print("✓ ComfyUI 安装完成")


def install_ffmpeg():
    """安装 ffmpeg"""
    print("\n" + "="*60)
    print("步骤 4: 安装 ffmpeg")
    print("="*60)
    
    result = run_cmd_output("which ffmpeg")
    if result:
        print(f"✓ ffmpeg 已安装: {result}")
        return
    
    run_cmd("sudo apt-get install -y ffmpeg")
    print("✓ ffmpeg 安装完成")


def install_comfyui_manager():
    """安装 ComfyUI Manager"""
    print("\n" + "="*60)
    print("步骤 5: 安装 ComfyUI Manager")
    print("="*60)
    
    custom_nodes_path = Path.home() / "ComfyUI" / "custom_nodes"
    manager_path = custom_nodes_path / "ComfyUI-Manager"
    
    if manager_path.exists():
        print(f"✓ ComfyUI Manager 已存在: {manager_path}")
    else:
        run_cmd("git clone https://github.com/ltdrdata/ComfyUI-Manager.git", cwd=str(custom_nodes_path))
    
    # 配置使用 pip（避免 uv 权限问题）
    manager_config_dir = Path.home() / "ComfyUI" / "user" / "__manager"
    manager_config_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = manager_config_dir / "config.ini"
    config_content = """[default]
use_uv = False
"""
    config_file.write_text(config_content)
    
    print(f"✓ Manager 配置已写入: {config_file}")
    print("✓ ComfyUI Manager 安装完成")


def install_tpu_custom_nodes():
    """安装 TPU Custom Nodes"""
    print("\n" + "="*60)
    print("步骤 6: 安装 TPU Custom Nodes")
    print("="*60)
    
    # 获取当前脚本所在目录（gpu-tpu-pedia/tpu/ComfyUI/）
    script_dir = Path(__file__).parent.resolve()
    source_nodes_path = script_dir / "custom_nodes"
    
    if not source_nodes_path.exists():
        print(f"[ERROR] 找不到 custom_nodes 目录: {source_nodes_path}")
        print("请确保从 gpu-tpu-pedia/tpu/ComfyUI/ 目录运行此脚本")
        sys.exit(1)
    
    dest_nodes_path = Path.home() / "ComfyUI" / "custom_nodes"
    
    nodes_to_copy = [
        "ComfyUI-CogVideoX-TPU",
        "ComfyUI-Wan2.1-TPU",
        "ComfyUI-Wan2.2-I2V-TPU",
        "ComfyUI-Flux.2-TPU",
        "ComfyUI-Crystools",
    ]
    
    for node in nodes_to_copy:
        src = source_nodes_path / node
        dst = dest_nodes_path / node
        
        if not src.exists():
            print(f"[WARNING] 源目录不存在: {src}")
            continue
        
        if dst.exists():
            print(f"  ✓ {node} 已存在，跳过")
        else:
            shutil.copytree(src, dst)
            print(f"  ✓ 复制: {node}")
    
    # 安装 Crystools 依赖
    crystools_req = dest_nodes_path / "ComfyUI-Crystools" / "requirements.txt"
    if crystools_req.exists():
        run_cmd(f"pip install -r {crystools_req}")
    
    # 复制示例输入图片到 ComfyUI/input/
    input_path = Path.home() / "ComfyUI" / "input"
    input_path.mkdir(parents=True, exist_ok=True)
    
    example_images = [
        ("ComfyUI-Wan2.2-I2V-TPU", "examples/wan_i2v_input.JPG"),
    ]
    
    for node, image_path in example_images:
        src = dest_nodes_path / node / image_path
        dst = input_path / Path(image_path).name
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)
            print(f"  ✓ 复制示例图片: {Path(image_path).name}")

    print("✓ TPU Custom Nodes 安装完成")


def install_tpu_dependencies():
    """安装 TPU 环境核心依赖"""
    print("\n" + "="*60)
    print("步骤 7: 安装 TPU 环境核心依赖")
    print("="*60)
    
    dependencies = [
        # 核心依赖
        "huggingface-hub",
        "transformers",
        "datasets",
        "evaluate",
        "accelerate",
        "timm",
        "flax",
        "numpy",
        "torchax",
        "tensorflow-cpu",
        
        # 辅助工具
        "sentencepiece",
        "imageio[ffmpeg]",
        "tpu-info",
        "matplotlib",
        
        # Flux.2 需要 jinja2 3.1.0+
        "jinja2>=3.1.0",
        
        # Wan2.1 需要 ftfy
        "ftfy",
    ]
    
    for dep in dependencies:
        run_cmd(f"pip install {dep}")
    
    # 安装 JAX with TPU support (0.8.1 版本，避免 CPU AOT 兼容性问题)
    run_cmd("pip install 'jax[tpu]==0.8.1' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html")
    
    print("✓ TPU 核心依赖安装完成")


def configure_environment():
    """配置环境变量"""
    print("\n" + "="*60)
    print("步骤 8: 配置环境变量")
    print("="*60)
    
    bashrc_path = Path.home() / ".bashrc"
    
    # 检查是否已配置
    bashrc_content = bashrc_path.read_text() if bashrc_path.exists() else ""
    
    marker = "# === ComfyUI TPU Environment ==="
    if marker in bashrc_content:
        print("✓ 环境变量已配置，跳过")
        return
    
    env_config = f"""

{marker}
# Python pip 用户目录
export PATH=$HOME/.local/bin:$PATH

# HuggingFace 配置
export HF_HOME=/dev/shm

# JAX 编译缓存（加速重复运行）
export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_cache
# === End ComfyUI TPU Environment ===
"""
    
    with open(bashrc_path, "a") as f:
        f.write(env_config)
    
    print(f"✓ 环境变量已添加到: {bashrc_path}")
    print("")
    print("注意: 请设置 HF_TOKEN 环境变量以访问 gated 模型:")
    print("  export HF_TOKEN=<your_huggingface_token>")
    print("  或添加到 ~/.bashrc 中")


def install_diffusers_tpu():
    """安装 diffusers-tpu（TPU 优化版 Diffusers）"""
    print("\n" + "="*60)
    print("步骤 9: 安装 diffusers-tpu")
    print("="*60)
    
    diffusers_path = Path.home() / "diffusers-tpu"
    
    if diffusers_path.exists():
        print(f"✓ diffusers-tpu 已存在: {diffusers_path}")
        # 更新并重新安装
        run_cmd("git pull", cwd=str(diffusers_path))
    else:
        run_cmd("git clone https://github.com/yangwhale/diffusers-tpu.git", cwd=str(Path.home()))
    
    run_cmd("pip install -e .", cwd=str(diffusers_path))
    
    print("✓ diffusers-tpu 安装完成")


def verify_installation():
    """验证安装"""
    print("\n" + "="*60)
    print("步骤 10: 验证安装")
    print("="*60)
    
    # 检查 Python 版本
    python_version = run_cmd_output("python --version")
    print(f"  Python: {python_version}")
    
    # 检查 JAX
    try:
        import jax
        print(f"  JAX: {jax.__version__}")
        devices = jax.devices()
        print(f"  TPU 设备: {len(devices)} 个")
        for i, d in enumerate(devices[:4]):  # 只显示前 4 个
            print(f"    {i}: {d}")
        if len(devices) > 4:
            print(f"    ... 还有 {len(devices) - 4} 个设备")
    except Exception as e:
        print(f"  [WARNING] JAX 检查失败: {e}")
    
    # 检查 ComfyUI
    comfyui_path = Path.home() / "ComfyUI"
    if comfyui_path.exists():
        print(f"  ComfyUI: ✓ {comfyui_path}")
    else:
        print(f"  ComfyUI: ✗ 未找到")
    
    # 检查 Custom Nodes
    custom_nodes = [
        "ComfyUI-Manager",
        "ComfyUI-CogVideoX-TPU",
        "ComfyUI-Wan2.1-TPU",
        "ComfyUI-Wan2.2-I2V-TPU",
        "ComfyUI-Flux.2-TPU",
        "ComfyUI-Crystools",
    ]
    
    nodes_path = comfyui_path / "custom_nodes"
    for node in custom_nodes:
        node_path = nodes_path / node
        status = "✓" if node_path.exists() else "✗"
        print(f"  {node}: {status}")


def main():
    """主函数"""
    print("="*60)
    print("ComfyUI on TPU - 一键安装脚本")
    print("="*60)
    print("")
    print("此脚本将安装以下组件:")
    print("  1. Python 3.12")
    print("  2. ComfyUI")
    print("  3. ComfyUI Manager")
    print("  4. TPU Custom Nodes (Flux, Wan2.1, Wan2.2-I2V, CogVideoX)")
    print("  5. TPU 核心依赖 (JAX 0.8.1, libtpu 0.0.30)")
    print("  6. diffusers-tpu")
    print("")
    
    # 确认继续
    if len(sys.argv) < 2 or sys.argv[1] != "-y":
        response = input("是否继续? [y/N] ")
        if response.lower() != "y":
            print("已取消安装")
            sys.exit(0)
    
    # 执行安装步骤
    install_python312()
    configure_pip()
    install_comfyui()
    install_ffmpeg()
    install_comfyui_manager()
    install_tpu_custom_nodes()
    install_tpu_dependencies()
    configure_environment()
    install_diffusers_tpu()
    verify_installation()
    
    print("\n" + "="*60)
    print("安装完成!")
    print("="*60)
    print("")
    print("下一步:")
    print("  1. 重新登录或执行: source ~/.bashrc")
    print("  2. 设置 HuggingFace Token (用于访问 gated 模型):")
    print("     export HF_TOKEN=<your_token>")
    print("  3. 启动 ComfyUI:")
    print("     cd ~/ComfyUI && python main.py --cpu --listen 0.0.0.0")
    print("")
    print("访问 ComfyUI: http://<VM_IP>:8188")


if __name__ == "__main__":
    main()
