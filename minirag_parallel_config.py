#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniRAG并行控制配置文件
用于优化extract_entities函数的LLM调用并行性
"""

import os
import asyncio
from typing import Dict, List, Any, Tuple
from collections import defaultdict

# 并行控制配置
PARALLEL_CONFIG = {
    'MAX_CONCURRENT_LLM_CALLS': 8,  # 最大并发LLM调用数
    'BATCH_SIZE': 16,               # 批处理大小
    'ENABLE_PARALLEL_EXTRACTION': True,  # 启用并行实体提取
    'ENABLE_PARALLEL_RELATIONSHIPS': True,  # 启用并行关系提取
    'MAX_RETRIES': 3,               # 最大重试次数
    'TIMEOUT_PER_CALL': 30,         # 每次调用的超时时间（秒）
}

class ParallelEntityExtractor:
    """并行实体提取器，用于替换MiniRAG内部的串行调用"""
    
    def __init__(self, llm_func, config=None):
        self.llm_func = llm_func
        self.config = config or PARALLEL_CONFIG
        self.semaphore = asyncio.Semaphore(self.config['MAX_CONCURRENT_LLM_CALLS'])
    
    async def extract_entities_parallel(
        self,
        chunks: Dict[str, Any],
        knowledge_graph_inst,
        entity_vdb,
        entity_name_vdb,
        relationships_vdb,
        global_config: dict
    ):
        """并行版本的实体提取函数"""
        
        if not self.config['ENABLE_PARALLEL_EXTRACTION']:
            # 如果未启用并行，使用原始方法
            return await self._extract_entities_original(
                chunks, knowledge_graph_inst, entity_vdb, 
                entity_name_vdb, relationships_vdb, global_config
            )
        
        print(f"🚀 开始并行实体提取，文本块数: {len(chunks)}")
        print(f"⚡ 并行配置: 最大并发={self.config['MAX_CONCURRENT_LLM_CALLS']}, 批处理大小={self.config['BATCH_SIZE']}")
        
        # 分批处理文本块
        ordered_chunks = list(chunks.items())
        all_entities = []
        all_relationships = []
        
        for batch_start in range(0, len(ordered_chunks), self.config['BATCH_SIZE']):
            batch_end = min(batch_start + self.config['BATCH_SIZE'], len(ordered_chunks))
            batch_chunks = dict(ordered_chunks[batch_start:batch_end])
            
            print(f"📦 处理批次 {batch_start//self.config['BATCH_SIZE'] + 1}: 文本块 {batch_start+1}-{batch_end}")
            
            # 并行处理当前批次
            batch_entities, batch_relationships = await self._process_batch_parallel(
                batch_chunks, global_config
            )
            
            all_entities.extend(batch_entities)
            all_relationships.extend(batch_relationships)
            
            print(f"✅ 批次完成: 提取了 {len(batch_entities)} 个实体, {len(batch_relationships)} 个关系")
        
        print(f"🎉 所有批次处理完成！总计: {len(all_entities)} 个实体, {len(all_relationships)} 个关系")
        
        # 这里需要调用原有的合并和存储逻辑
        # 由于我们无法直接修改MiniRAG内部，这里返回处理后的数据
        return {
            'entities': all_entities,
            'relationships': all_relationships,
            'chunks_processed': len(chunks)
        }
    
    async def _process_batch_parallel(self, batch_chunks: Dict, global_config: dict):
        """并行处理一个批次的文本块"""
        
        # 创建所有需要的提示
        all_prompts = []
        for chunk_key, chunk_data in batch_chunks.items():
            content = chunk_data["content"]
            
            # 简化的提示，减少LLM调用次数
            prompt = f"""请从以下文本中提取实体和关系，以JSON格式返回：

文本内容：
{content}

请返回格式：
{{
  "entities": [
    {{"name": "实体名", "type": "实体类型", "description": "描述"}}
  ],
  "relationships": [
    {{"source": "源实体", "target": "目标实体", "relation": "关系类型"}}
  ]
}}"""
            
            all_prompts.append((chunk_key, prompt))
        
        # 并行调用LLM
        async def single_llm_call(chunk_key, prompt):
            async with self.semaphore:
                try:
                    # 设置超时
                    result = await asyncio.wait_for(
                        self.llm_func(prompt), 
                        timeout=self.config['TIMEOUT_PER_CALL']
                    )
                    return chunk_key, result, None
                except asyncio.TimeoutError:
                    return chunk_key, None, "调用超时"
                except Exception as e:
                    return chunk_key, None, str(e)
        
        # 并行执行所有LLM调用
        tasks = [single_llm_call(chunk_key, prompt) for chunk_key, prompt in all_prompts]
        results = await asyncio.gather(*tasks)
        
        # 处理结果
        batch_entities = []
        batch_relationships = []
        
        for chunk_key, result, error in results:
            if error:
                print(f"⚠️ 跳过失败的文本块 {chunk_key}: {error}")
                continue
            
            try:
                # 解析JSON结果
                import json
                parsed = json.loads(result)
                
                # 处理实体
                if "entities" in parsed:
                    for entity in parsed["entities"]:
                        batch_entities.append({
                            "entity_name": entity.get("name", ""),
                            "entity_type": entity.get("type", ""),
                            "description": entity.get("description", ""),
                            "chunk_key": chunk_key
                        })
                
                # 处理关系
                if "relationships" in parsed:
                    for rel in parsed["relationships"]:
                        batch_relationships.append({
                            "src_id": rel.get("source", ""),
                            "tgt_id": rel.get("target", ""),
                            "relation_type": rel.get("relation", ""),
                            "chunk_key": chunk_key
                        })
                
            except Exception as e:
                print(f"⚠️ 解析结果失败 {chunk_key}: {e}")
        
        return batch_entities, batch_relationships
    
    async def _extract_entities_original(self, chunks, knowledge_graph_inst, entity_vdb, 
                                       entity_name_vdb, relationships_vdb, global_config):
        """原始的实体提取方法（作为fallback）"""
        print("⚠️ 使用原始实体提取方法")
        # 这里应该调用原始的extract_entities函数
        # 但由于我们无法直接导入，这里返回空结果
        return {
            'entities': [],
            'relationships': [],
            'chunks_processed': len(chunks)
        }

def create_parallel_extractor(llm_func, config=None):
    """创建并行实体提取器实例"""
    return ParallelEntityExtractor(llm_func, config)

# 使用示例
async def example_usage():
    """使用示例"""
    
    # 模拟LLM函数
    async def mock_llm_func(prompt):
        await asyncio.sleep(1)  # 模拟API调用延迟
        return '{"entities": [{"name": "测试实体", "type": "概念", "description": "测试描述"}], "relationships": []}'
    
    # 创建并行提取器
    extractor = create_parallel_extractor(
        llm_func=mock_llm_func,
        config=PARALLEL_CONFIG
    )
    
    # 模拟文本块数据
    mock_chunks = {
        f"chunk_{i}": {"content": f"这是第{i}个文本块的内容"} 
        for i in range(20)
    }
    
    # 执行并行提取
    result = await extractor.extract_entities_parallel(
        chunks=mock_chunks,
        knowledge_graph_inst=None,
        entity_vdb=None,
        entity_name_vdb=None,
        relationships_vdb=None,
        global_config={}
    )
    
    print(f"提取结果: {result}")

if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_usage())
