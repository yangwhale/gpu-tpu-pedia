#!/bin/bash

# 简化版自动查找pybind11并编译项目的脚本
# 使用方法: ./build.sh [clean]

# 错误处理函数
handle_error() {
    echo "错误: $1"
    exit 1
}

# 检查必要的依赖
check_dependencies() {
    echo "检查依赖..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        handle_error "未找到python3，请安装Python 3.6或更高版本"
    fi
    
    # 检查CMake
    if ! command -v cmake &> /dev/null; then
        echo "安装CMake..."
        apt-get update && apt-get install -y cmake
    fi
    
    # 检查bc命令(用于版本比较)
    if ! command -v bc &> /dev/null; then
        echo "安装bc命令行工具..."
        apt-get update && apt-get install -y bc
    fi
    
    # 检查pip
    if ! command -v pip &> /dev/null; then
        echo "安装pip..."
        apt-get update && apt-get install -y python3-pip
    fi
    
    # 检查pybind11
    if ! pip list | grep -q pybind11; then
        echo "安装pybind11..."
        pip install pybind11
    fi
}

# 查找pybind11
find_pybind11() {
    echo "查找pybind11..."
    
    # 使用Python查找pybind11的CMake目录
    PYBIND11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null)
    
    if [ -n "$PYBIND11_DIR" ] && [ -d "$PYBIND11_DIR" ]; then
        echo "找到pybind11: $PYBIND11_DIR"
        return 0
    fi
    
    # 尝试使用pip查找
    PIP_LOCATION=$(pip show pybind11 2>/dev/null | grep -i "location:" | awk '{print $2}')
    
    if [ -n "$PIP_LOCATION" ]; then
        POSSIBLE_DIR="$PIP_LOCATION/pybind11/share/cmake/pybind11"
        if [ -d "$POSSIBLE_DIR" ]; then
            echo "找到pybind11: $POSSIBLE_DIR"
            PYBIND11_DIR="$POSSIBLE_DIR"
            return 0
        fi
    fi
    
    return 1
}

# 编译项目
build_project() {
    # 清理构建目录(如果需要)
    if [ "$1" = "clean" ]; then
        echo "清理构建目录..."
        rm -rf build
    fi
    
    # 创建构建目录
    mkdir -p build
    cd build || handle_error "无法进入构建目录"
    
    # 运行CMake
    echo "运行CMake配置..."
    if [ -n "$PYBIND11_DIR" ]; then
        cmake -Dpybind11_DIR="$PYBIND11_DIR" ..
    else
        cmake ..
    fi
    
    if [ $? -ne 0 ]; then
        handle_error "CMake配置失败"
    fi
    
    # 编译项目
    echo "开始编译..."
    make -j$(nproc)
    
    if [ $? -ne 0 ]; then
        handle_error "编译失败"
    fi
    
    echo "编译成功! 可执行文件位于: $(pwd)/cublas_backend.so"
}

# 主函数
main() {
    echo "==== 开始编译 CublasBenchmark ===="
    
    # 检查依赖
    check_dependencies
    
    # 查找pybind11
    find_pybind11
    if [ $? -ne 0 ]; then
        echo "警告: 未找到pybind11，尝试直接运行CMake..."
        echo "如果CMake失败，请手动安装pybind11: pip install pybind11"
        PYBIND11_DIR=""
    fi
    
    # 编译项目
    build_project "$1"
    
    echo "==== 编译完成 ===="
}

# 执行主函数
main "$@"

