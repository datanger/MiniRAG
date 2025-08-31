#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniRAG 终端问答CLI程序
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional
import logging
from dotenv import load_dotenv

from rag_client import MiniRAGClientSync

# 加载环境变量
# 从项目根目录加载config.env
project_root = Path(__file__).parent.parent
config_path = project_root / "config.env"
load_dotenv(config_path)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGCLI:
    """MiniRAG 终端问答CLI类"""
    
    def __init__(self, server_url: str = None, api_key: Optional[str] = None):
        # 完全从config.env读取配置
        if server_url is None:
            host = os.getenv("HOST")
            port = os.getenv("PORT")
            if not host or not port:
                raise ValueError("HOST和PORT必须在config.env中配置")
            server_url = f"http://{host}:{port}"
        
        self.server_url = server_url
        self.api_key = api_key
        self.client = MiniRAGClientSync(server_url, api_key)
        self.conversation_history = []
        self.max_history = 5
        
    def print_banner(self):
        """打印欢迎横幅"""
        print("=" * 60)
        print("🚀 MiniRAG 终端问答系统")
        print("=" * 60)
        print(f"服务器地址: {self.server_url}")
        print(f"API密钥: {'已设置' if self.api_key else '未设置'}")
        print("=" * 60)
        print("支持的命令:")
        print("  /help     - 显示帮助信息")
        print("  /health   - 检查服务器状态")
        print("  /docs     - 显示已索引文档")
        print("  /scan     - 扫描并索引新文档")
        print("  /upload   - 上传文档")
        print("  /insert   - 插入文本")
        print("  /clear    - 清空对话历史")
        print("  /quit     - 退出程序")
        print("=" * 60)
    
    def check_server_health(self) -> bool:
        """检查服务器健康状态"""
        try:
            print("🔍 检查服务器状态...")
            health = self.client.health_check()
            
            if health.get("status") == "healthy":
                print("✅ 服务器运行正常")
                config = health.get("configuration", {})
                print(f"   - LLM绑定: {config.get('llm_binding', 'N/A')}")
                print(f"   - 嵌入模型: {config.get('embedding_model', 'N/A')}")
                print(f"   - 已索引文档: {health.get('indexed_files_count', 0)}")
                return True
            else:
                print("❌ 服务器状态异常")
                return False
                
        except Exception as e:
            print(f"❌ 无法连接到服务器: {e}")
            return False
    
    def show_documents(self):
        """显示已索引的文档"""
        try:
            print("📚 获取已索引文档...")
            docs = self.client.get_documents()
            
            if isinstance(docs, list) and docs:
                print(f"✅ 找到 {len(docs)} 个已索引文档:")
                for i, doc in enumerate(docs, 1):
                    print(f"  {i}. {doc}")
            else:
                print("ℹ️  暂无已索引文档")
                
        except Exception as e:
            print(f"❌ 获取文档列表失败: {e}")
    
    def scan_documents(self):
        """扫描并索引新文档"""
        try:
            print("🔍 扫描并索引新文档...")
            result = self.client.scan_documents()
            
            if result.get("status") == "success":
                print(f"✅ 扫描完成")
                print(f"   - 索引文档数: {result.get('indexed_count', 0)}")
                print(f"   - 总文档数: {result.get('total_documents', 0)}")
            else:
                print(f"❌ 扫描失败: {result.get('message', '未知错误')}")
                
        except Exception as e:
            print(f"❌ 扫描文档失败: {e}")
    
    def upload_document(self, file_path: str, description: str = ""):
        """上传文档"""
        try:
            if not os.path.exists(file_path):
                print(f"❌ 文件不存在: {file_path}")
                return
            
            print(f"📤 上传文档: {file_path}")
            result = self.client.upload_document(file_path, description)
            
            if result.get("status") == "success":
                print(f"✅ 文档上传成功")
                print(f"   - 文件名: {result.get('message', 'N/A')}")
                print(f"   - 总文档数: {result.get('total_documents', 0)}")
            else:
                print(f"❌ 文档上传失败: {result.get('message', '未知错误')}")
                
        except Exception as e:
            print(f"❌ 上传文档失败: {e}")
    
    def insert_text(self, text: str, description: str = ""):
        """插入文本"""
        try:
            print("📝 插入文本到RAG系统...")
            result = self.client.insert_text(text, description)
            
            if result.get("status") == "success":
                print(f"✅ 文本插入成功")
                print(f"   - 消息: {result.get('message', 'N/A')}")
                print(f"   - 文档数: {result.get('document_count', 0)}")
            else:
                print(f"❌ 文本插入失败: {result.get('message', '未知错误')}")
                
        except Exception as e:
            print(f"❌ 插入文本失败: {e}")
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history.clear()
        print("🗑️  对话历史已清空")
    
    def add_to_history(self, question: str, answer: str):
        """添加对话到历史"""
        self.conversation_history.append({
            "question": question,
            "content": answer
        })
        
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
    
    def query_rag(self, question: str, mode: str = "mini") -> str:
        """查询RAG系统"""
        try:
            print(f"🤖 正在查询 (模式: {mode})...")
            result = self.client.query(question, mode)
            
            if "response" in result:
                answer = result["response"]
                print(f"✅ 查询完成")
                return answer
            else:
                error_msg = result.get("message", "未知错误")
                print(f"❌ 查询失败: {error_msg}")
                return f"查询失败: {error_msg}"
                
        except Exception as e:
            error_msg = f"查询异常: {e}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def process_command(self, command: str) -> bool:
        """处理特殊命令"""
        command = command.strip().lower()
        
        if command == "/help":
            print("📚 帮助信息:")
            print("  /help     - 显示帮助信息")
            print("  /health   - 检查服务器状态")
            print("  /docs     - 显示已索引文档")
            print("  /scan     - 扫描并索引新文档")
            print("  /upload   - 上传文档")
            print("  /insert   - 插入文本")
            print("  /clear    - 清空对话历史")
            print("  /quit     - 退出程序")
            return True
        elif command == "/health":
            self.check_server_health()
            return True
        elif command == "/docs":
            self.show_documents()
            return True
        elif command == "/scan":
            self.scan_documents()
            return True
        elif command == "/clear":
            self.clear_history()
            return True
        elif command == "/quit" or command == "/exit":
            print("👋 再见！")
            return False
        elif command.startswith("/upload "):
            parts = command.split(" ", 2)
            if len(parts) >= 2:
                file_path = parts[1]
                description = parts[2] if len(parts) > 2 else ""
                self.upload_document(file_path, description)
            else:
                print("❌ 用法: /upload <文件路径> [描述]")
            return True
        elif command.startswith("/insert "):
            parts = command.split(" ", 2)
            if len(parts) >= 2:
                text = parts[1]
                description = parts[2] if len(parts) > 2 else ""
                self.insert_text(text, description)
            else:
                print("❌ 用法: /insert <文本内容> [描述]")
            return True
        
        return False
    
    def run(self):
        """运行CLI主循环"""
        if not self.check_server_health():
            print("❌ 无法连接到MiniRAG服务器，请检查服务器是否运行")
            return
        
        self.print_banner()
        current_mode = "mini"
        
        while True:
            try:
                user_input = input(f"\n💭 [{current_mode}] 请输入问题或命令: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith("/"):
                    if not self.process_command(user_input):
                        break
                    continue
                
                print(f"\n🔍 问题: {user_input}")
                print("-" * 50)
                
                answer = self.query_rag(user_input, current_mode)
                
                if answer and not answer.startswith("查询失败"):
                    print(f"\n💡 回答: {answer}")
                    self.add_to_history(user_input, answer)
                else:
                    print(f"\n❌ {answer}")
                
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n\n👋 程序被中断，再见！")
                break
            except EOFError:
                print("\n\n👋 程序结束，再见！")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                logger.error(f"CLI运行错误: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MiniRAG 终端问答CLI")
    parser.add_argument(
        "--server", 
        help="MiniRAG服务器地址 (默认: 从config.env自动读取)"
    )
    parser.add_argument(
        "--api-key",
        help="API密钥 (默认: 从config.env自动读取)"
    )
    
    args = parser.parse_args()
    
    # 优先使用命令行参数，否则自动从config.env读取
    server_url = args.server
    api_key = args.api_key
    
    cli = RAGCLI(server_url, api_key)
    
    try:
        cli.run()
    except Exception as e:
        print(f"❌ CLI运行失败: {e}")
        logger.error(f"CLI运行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
