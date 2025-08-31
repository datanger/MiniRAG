#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试RAG查询功能的脚本
"""

import requests
import json

def test_rag_query():
    """测试RAG查询功能"""
    
    # 服务器地址
    base_url = "http://localhost:9721"
    
    # 测试查询
    test_queries = [
        "什么是MiniRAG？",
        "VPN访问指南的主要内容是什么？",
        "智能体有哪些特点？",
        "Block-NeRF是什么技术？"
    ]
    
    print("🧪 开始测试RAG查询功能...")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 测试查询 {i}: {query}")
        print("-" * 30)
        
        try:
            # 发送查询请求
            response = requests.post(
                f"{base_url}/query",
                json={
                    "query": query,
                    "mode": "light",
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "无回答")
                print(f"✅ 查询成功")
                print(f"📄 回答: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            else:
                print(f"❌ 查询失败: HTTP {response.status_code}")
                print(f"错误信息: {response.text}")
                
        except requests.exceptions.Timeout:
            print("⏰ 查询超时")
        except requests.exceptions.ConnectionError:
            print("🔌 连接错误，请检查服务器是否运行")
        except Exception as e:
            print(f"❌ 查询异常: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🎯 测试完成！")

if __name__ == "__main__":
    test_rag_query()
