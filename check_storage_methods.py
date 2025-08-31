#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查存储组件的实际可用方法
"""

import sys
import os
sys.path.append('.')

def check_storage_methods():
    """检查存储组件的方法"""
    print("🔍 检查存储组件的方法...")
    
    try:
        from minirag import MiniRAG
        
        # 创建MiniRAG实例
        working_dir = "./rag_storage"
        
        # 模拟函数
        def mock_llm_func(prompt, **kwargs):
            return "这是一个模拟的LLM响应"
        
        class MockEmbeddingFunc:
            def __init__(self, embedding_dim=1024):
                self.embedding_dim = embedding_dim
            
            def __call__(self, texts):
                return [[0.1] * self.embedding_dim] * len(texts)
        
        mock_embed_func = MockEmbeddingFunc(embedding_dim=1024)
        
        rag = MiniRAG(
            working_dir=working_dir,
            llm_model_func=mock_llm_func,
            llm_model_name="test-model",
            embedding_func=mock_embed_func,
            chunk_token_size=1200,
            chunk_overlap_token_size=100,
            tiktoken_model_name="cl100k_base"
        )
        
        print("✅ MiniRAG实例创建成功")
        
        # 检查各个存储组件的方法
        print("\n📊 检查存储组件方法...")
        
        # 检查 text_chunks
        if hasattr(rag, 'text_chunks'):
            print(f"\n📝 text_chunks ({type(rag.text_chunks).__name__}):")
            methods = [method for method in dir(rag.text_chunks) if not method.startswith('_')]
            print(f"   可用方法: {methods}")
            
            # 尝试调用可能的方法
            if 'get' in methods:
                try:
                    result = rag.text_chunks.get("test")
                    print(f"   ✅ get() 方法可用")
                except Exception as e:
                    print(f"   ❌ get() 方法调用失败: {e}")
            
            if 'keys' in methods:
                try:
                    result = rag.text_chunks.keys()
                    print(f"   ✅ keys() 方法可用")
                except Exception as e:
                    print(f"   ❌ keys() 方法调用失败: {e}")
        
        # 检查 doc_status
        if hasattr(rag, 'doc_status'):
            print(f"\n📊 doc_status ({type(rag.doc_status).__name__}):")
            methods = [method for method in dir(rag.doc_status) if not method.startswith('_')]
            print(f"   可用方法: {methods}")
            
            # 尝试调用可能的方法
            if 'get' in methods:
                try:
                    result = rag.doc_status.get("test")
                    print(f"   ✅ get() 方法可用")
                except Exception as e:
                    print(f"   ❌ get() 方法调用失败: {e}")
            
            if 'keys' in methods:
                try:
                    result = rag.doc_status.keys()
                    print(f"   ✅ keys() 方法可用")
                except Exception as e:
                    print(f"   ❌ keys() 方法调用失败: {e}")
        
        # 检查 full_docs
        if hasattr(rag, 'full_docs'):
            print(f"\n📄 full_docs ({type(rag.full_docs).__name__}):")
            methods = [method for method in dir(rag.full_docs) if not method.startswith('_')]
            print(f"   可用方法: {methods}")
            
            # 尝试调用可能的方法
            if 'get' in methods:
                try:
                    result = rag.full_docs.get("test")
                    print(f"   ✅ get() 方法可用")
                except Exception as e:
                    print(f"   ❌ get() 方法调用失败: {e}")
            
            if 'keys' in methods:
                try:
                    result = rag.full_docs.keys()
                    print(f"   ✅ keys() 方法可用")
                except Exception as e:
                    print(f"   ❌ keys() 方法调用失败: {e}")
        
        # 尝试获取实际数据
        print("\n🧪 尝试获取实际数据...")
        
        try:
            # 尝试获取文档状态
            if hasattr(rag.doc_status, 'keys'):
                doc_keys = list(rag.doc_status.keys())
                print(f"✅ 获取到 {len(doc_keys)} 个文档键")
                if doc_keys:
                    print(f"   第一个文档键: {doc_keys[0]}")
                    
                    # 尝试获取第一个文档的状态
                    doc_status = rag.doc_status.get(doc_keys[0])
                    print(f"   文档状态: {doc_status}")
            
            # 尝试获取文本块
            if hasattr(rag.text_chunks, 'keys'):
                chunk_keys = list(rag.text_chunks.keys())
                print(f"✅ 获取到 {len(chunk_keys)} 个文本块键")
                if chunk_keys:
                    print(f"   第一个文本块键: {chunk_keys[0]}")
                    
                    # 尝试获取第一个文本块
                    chunk_data = rag.text_chunks.get(chunk_keys[0])
                    print(f"   文本块数据: {type(chunk_data)}")
                    
        except Exception as e:
            print(f"❌ 获取实际数据失败: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ 检查存储组件方法失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🔍 检查存储组件的实际可用方法")
    print("=" * 60)
    
    check_storage_methods()
    
    print("\n" + "=" * 60)
    print("🎯 检查完成！")

if __name__ == "__main__":
    main()
