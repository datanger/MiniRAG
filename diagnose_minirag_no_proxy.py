#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断MiniRAG系统状态的脚本（绕过代理）
"""

import requests
import json
import os

def check_system_health():
    """检查系统健康状态"""
    print("🏥 检查系统健康状态...")
    
    # 创建不使用代理的session
    session = requests.Session()
    session.trust_env = False  # 不使用环境变量中的代理设置
    
    try:
        response = session.get("http://localhost:9721/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ 系统状态: {health_data.get('status', 'unknown')}")
            print(f"📁 工作目录: {health_data.get('working_directory', 'unknown')}")
            print(f"📂 输入目录: {health_data.get('input_directory', 'unknown')}")
            print(f"📊 索引文件数: {health_data.get('indexed_files_count', 0)}")
            print(f"🤖 LLM绑定: {health_data.get('configuration', {}).get('llm_binding', 'unknown')}")
            print(f"🔗 LLM主机: {health_data.get('configuration', {}).get('llm_binding_host', 'unknown')}")
            print(f"📝 LLM模型: {health_data.get('configuration', {}).get('llm_model', 'unknown')}")
            return True
        else:
            print(f"❌ 健康检查失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 健康检查异常: {e}")
        return False

def check_storage_files():
    """检查存储文件状态"""
    print("\n💾 检查存储文件状态...")
    
    working_dir = "./rag_storage"
    if not os.path.exists(working_dir):
        print(f"❌ 工作目录不存在: {working_dir}")
        return False
    
    storage_files = [
        "kv_store_doc_status.json",
        "kv_store_full_docs.json", 
        "kv_store_text_chunks.json",
        "vdb_entities.json",
        "vdb_relationships.json"
    ]
    
    for file_name in storage_files:
        file_path = os.path.join(working_dir, file_name)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_name}: {size} bytes")
        else:
            print(f"❌ {file_name}: 不存在")
    
    return True

def check_documents_endpoint():
    """检查文档端点"""
    print("\n📚 检查文档端点...")
    
    session = requests.Session()
    session.trust_env = False
    
    try:
        response = session.get("http://localhost:9721/documents", timeout=10)
        if response.status_code == 200:
            docs_data = response.json()
            if isinstance(docs_data, list):
                print(f"✅ 文档端点返回: {len(docs_data)} 个文档")
                if docs_data:
                    print(f"📄 第一个文档: {docs_data[0]}")
            else:
                print(f"⚠️ 文档端点返回类型异常: {type(docs_data)}")
                print(f"📄 返回内容: {docs_data}")
        else:
            print(f"❌ 文档端点失败: HTTP {response.status_code}")
            print(f"错误信息: {response.text}")
    except Exception as e:
        print(f"❌ 文档端点异常: {e}")

def check_text_insertion():
    """测试文本插入功能"""
    print("\n📝 测试文本插入功能...")
    
    session = requests.Session()
    session.trust_env = False
    
    try:
        test_text = "这是一个测试文本，用于验证MiniRAG系统是否正常工作。"
        response = session.post(
            "http://localhost:9721/documents/text",
            json={"text": test_text, "description": "测试插入"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 文本插入成功: {result.get('status', 'unknown')}")
            print(f"📊 文档计数: {result.get('document_count', 0)}")
        else:
            print(f"❌ 文本插入失败: HTTP {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"❌ 文本插入异常: {e}")

def check_vector_search():
    """检查向量搜索功能"""
    print("\n🔍 检查向量搜索功能...")
    
    # 检查向量数据库文件
    working_dir = "./rag_storage"
    vdb_files = ["vdb_entities.json", "vdb_relationships.json"]
    
    for file_name in vdb_files:
        file_path = os.path.join(working_dir, file_name)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        print(f"✅ {file_name}: {len(data)} 个向量")
                    else:
                        print(f"⚠️ {file_name}: 数据格式异常")
            except Exception as e:
                print(f"❌ {file_name}: 读取失败 - {e}")
        else:
            print(f"❌ {file_name}: 不存在")

def main():
    """主函数"""
    print("🔍 MiniRAG系统诊断开始（绕过代理）")
    print("=" * 60)
    
    # 检查系统健康状态
    if not check_system_health():
        print("❌ 系统健康检查失败，停止诊断")
        return
    
    # 检查存储文件
    check_storage_files()
    
    # 检查文档端点
    check_documents_endpoint()
    
    # 检查向量搜索
    check_vector_search()
    
    # 测试文本插入
    check_text_insertion()
    
    print("\n" + "=" * 60)
    print("🎯 诊断完成！")

if __name__ == "__main__":
    main()
