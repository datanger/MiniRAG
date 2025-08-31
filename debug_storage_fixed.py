#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试MiniRAG存储组件初始化问题（修复版）
"""

import sys
import os
sys.path.append('.')

def check_storage_components():
    """检查存储组件"""
    print("🔍 检查存储组件初始化...")
    
    try:
        from minirag import MiniRAG
        print("✅ MiniRAG模块导入成功")
        
        # 检查工作目录
        working_dir = "./rag_storage"
        if not os.path.exists(working_dir):
            print(f"❌ 工作目录不存在: {working_dir}")
            return
        
        print(f"✅ 工作目录存在: {working_dir}")
        
        # 尝试创建MiniRAG实例
        print("\n🔧 尝试创建MiniRAG实例...")
        
        # 模拟LLM函数
        def mock_llm_func(prompt, **kwargs):
            return "这是一个模拟的LLM响应"
        
        # 创建正确的embedding函数类
        class MockEmbeddingFunc:
            def __init__(self, embedding_dim=1024):
                self.embedding_dim = embedding_dim
            
            def __call__(self, texts):
                return [[0.1] * self.embedding_dim] * len(texts)
        
        mock_embed_func = MockEmbeddingFunc(embedding_dim=1024)
        
        try:
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
            
            # 检查存储组件
            print("\n📊 检查存储组件状态...")
            
            if hasattr(rag, 'text_chunks'):
                print(f"✅ text_chunks: {type(rag.text_chunks)}")
                if hasattr(rag.text_chunks, 'get_all'):
                    print("✅ text_chunks 有 get_all 方法")
                else:
                    print("❌ text_chunks 缺少 get_all 方法")
            
            if hasattr(rag, 'doc_status'):
                print(f"✅ doc_status: {type(rag.doc_status)}")
                if hasattr(rag.doc_status, 'get_all'):
                    print("✅ doc_status 有 get_all 方法")
                else:
                    print("❌ doc_status 缺少 get_all 方法")
            
            if hasattr(rag, 'full_docs'):
                print(f"✅ full_docs: {type(rag.full_docs)}")
                if hasattr(rag.full_docs, 'get_all'):
                    print("✅ full_docs 有 get_all 方法")
                else:
                    print("❌ full_docs 缺少 get_all 方法")
            
            if hasattr(rag, 'entities_vdb'):
                print(f"✅ entities_vdb: {type(rag.entities_vdb)}")
            
            if hasattr(rag, 'relationships_vdb'):
                print(f"✅ relationships_vdb: {type(rag.relationships_vdb)}")
            
            # 测试存储组件方法
            print("\n🧪 测试存储组件方法...")
            
            try:
                if hasattr(rag.doc_status, 'get_all'):
                    doc_status_data = rag.doc_status.get_all()
                    print(f"✅ doc_status.get_all() 成功: {len(doc_status_data)} 项")
                
                if hasattr(rag.full_docs, 'get_all'):
                    full_docs_data = rag.full_docs.get_all()
                    print(f"✅ full_docs.get_all() 成功: {len(full_docs_data)} 项")
                
                if hasattr(rag.text_chunks, 'get_all'):
                    text_chunks_data = rag.text_chunks.get_all()
                    print(f"✅ text_chunks.get_all() 成功: {len(text_chunks_data)} 项")
                    
            except Exception as e:
                print(f"❌ 测试存储组件方法失败: {e}")
                
        except Exception as e:
            print(f"❌ MiniRAG实例创建失败: {e}")
            import traceback
            traceback.print_exc()
            
    except ImportError as e:
        print(f"❌ 导入MiniRAG失败: {e}")
    except Exception as e:
        print(f"❌ 检查存储组件异常: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🔍 深入调试MiniRAG存储组件（修复版）")
    print("=" * 60)
    
    check_storage_components()
    
    print("\n" + "=" * 60)
    print("🎯 调试完成！")

if __name__ == "__main__":
    main()
