#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动MiniRAG服务器的脚本
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
project_root = Path(__file__).parent.parent
config_path = project_root / "config.env"
load_dotenv(config_path)

def check_server_running(url: str = None) -> bool:
    """检查服务器是否正在运行"""
    try:
        if url is None:
            # 完全从config.env读取配置
            host = os.getenv("HOST")
            port = os.getenv("PORT")
            if not host or not port:
                raise ValueError("HOST和PORT必须在config.env中配置")
            url = f"http://{host}:{port}"
        
        # 增加超时时间到10秒，给服务器更多初始化时间
        response = requests.get(f"{url}/health", timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        # 只捕获网络相关异常，避免隐藏其他重要错误
        return False
    except Exception as e:
        # 记录其他异常以便调试
        print(f"⚠️  检查服务器状态时出现异常: {e}")
        return False

def start_server():
    """启动MiniRAG服务器"""
    print("🚀 启动MiniRAG服务器...")
    
    # 检查服务器是否已经在运行
    if check_server_running():
        print("✅ 服务器已经在运行")
        return True
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    server_script = project_root / "minirag" / "api" / "minirag_server.py"
    
    if not server_script.exists():
        print(f"❌ 找不到服务器脚本: {server_script}")
        return False
    
    try:
        # 启动服务器
        print(f"📁 服务器脚本路径: {server_script}")
        print("🔄 正在启动服务器...")
        
        # 从config.env自动读取所有配置，无需传入任何参数
        config_path = project_root / "config.env"
        if config_path.exists():
            print(f"📁 配置文件路径: {config_path}")
            print("✨ 所有配置将通过环境变量自动读取")
            # 使用Python启动服务器，完全依赖环境变量配置
            # 修改为前台运行，以便显示日志
            print("🔄 启动MiniRAG服务器...")
            print("=" * 60)
            # 直接运行服务器进程，不后台运行
            os.system(f"{sys.executable} {server_script}")
            return True
        else:
            print(f"⚠️  配置文件不存在: {config_path}")
            print("🔄 使用默认配置启动服务器...")
            # 使用config.env中的配置启动服务器
            host = os.getenv("HOST")
            port = os.getenv("PORT")
            working_dir = os.getenv("WORKING_DIR")
            input_dir = os.getenv("INPUT_DIR")
            
            if not all([host, port, working_dir, input_dir]):
                print("❌ 配置文件不完整，缺少必要的配置项")
                return False
            
            # 修改为前台运行，以便显示日志
            print("🔄 启动MiniRAG服务器...")
            print("=" * 60)
            # 直接运行服务器进程，不后台运行
            os.system(f"{sys.executable} {server_script} --host {host} --port {port} --working-dir {working_dir} --input-dir {input_dir}")
            return True
        
        # 服务器现在在前台运行，不需要等待逻辑
        # 如果执行到这里，说明服务器已经启动并运行
        return True
        
    except Exception as e:
        print(f"❌ 启动服务器失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🚀 MiniRAG 服务器启动器")
    print("=" * 60)
    
    if start_server():
        print("\n🎉 服务器启动完成!")
        print("💡 现在可以使用以下工具:")
        print("   - rag_cli.py: 终端问答CLI")
        print("   - rag_builder.py: RAG系统构建器")
        print("   - 浏览器访问: http://localhost:9721/docs")
    else:
        print("\n❌ 服务器启动失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
