#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniRAG并行控制补丁
用于优化extract_entities函数的LLM调用并行性
"""

import asyncio
import logging
from typing import Dict, List, Any, Tuple
from collections import defaultdict

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParallelEntityExtractor:
    """并行实体提取器，用于替换MiniRAG内部的串行调用"""
    
    def __init__(self, llm_func, max_concurrent=4, batch_size=8):
        self.llm_func = llm_func
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
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
        
        ordered_chunks = list(chunks.items())
        total_chunks = len(ordered_chunks)
        
        logger.info(f"🚀 开始并行实体提取，总文本块数: {total_chunks}")
        logger.info(f"⚡ 并行配置: 最大并发={self.max_concurrent}, 批处理大小={self.batch_size}")
        
        # 分批处理文本块
        all_entities = []
        all_relationships = []
        
        for batch_start in range(0, total_chunks, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_chunks)
            batch_chunks = ordered_chunks[batch_start:batch_end]
            
            logger.info(f"📦 处理批次 {batch_start//self.batch_size + 1}: 文本块 {batch_start+1}-{batch_end}")
            
            # 并行处理当前批次
            batch_results = await self._process_batch_parallel(batch_chunks, global_config)
            
            # 收集结果
            for entities, relationships in batch_results:
                all_entities.extend(entities)
                all_relationships.extend(relationships)
            
            logger.info(f"✅ 批次完成: 提取了 {len(all_entities)} 个实体, {len(all_relationships)} 个关系")
        
        logger.info(f"🎉 所有批次处理完成！总计: {len(all_entities)} 个实体, {len(all_relationships)} 个关系")
        
        # 这里需要调用原有的合并和存储逻辑
        # 由于我们无法直接修改MiniRAG内部，这里返回处理后的数据
        return {
            'entities': all_entities,
            'relationships': all_relationships,
            'chunks_processed': total_chunks
        }
    
    async def _process_batch_parallel(self, batch_chunks: List[Tuple], global_config: dict):
        """并行处理一个批次的文本块"""
        
        # 创建所有需要的提示
        all_prompts = []
        for chunk_key, chunk_data in batch_chunks:
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
                    result = await self.llm_func(prompt)
                    return chunk_key, result, None
                except Exception as e:
                    logger.warning(f"LLM调用失败 {chunk_key}: {e}")
                    return chunk_key, None, str(e)
        
        # 并行执行所有LLM调用
        tasks = [single_llm_call(chunk_key, prompt) for chunk_key, prompt in all_prompts]
        results = await asyncio.gather(*tasks)
        
        # 处理结果
        batch_results = []
        for chunk_key, result, error in results:
            if error:
                logger.warning(f"跳过失败的文本块 {chunk_key}: {error}")
                batch_results.append(([], []))
                continue
            
            try:
                # 解析JSON结果
                import json
                parsed = json.loads(result)
                
                entities = []
                relationships = []
                
                # 处理实体
                if "entities" in parsed:
                    for entity in parsed["entities"]:
                        entities.append({
                            "entity_name": entity.get("name", ""),
                            "entity_type": entity.get("type", ""),
                            "description": entity.get("description", ""),
                            "chunk_key": chunk_key
                        })
                
                # 处理关系
                if "relationships" in parsed:
                    for rel in parsed["relationships"]:
                        relationships.append({
                            "src_id": rel.get("source", ""),
                            "tgt_id": rel.get("target", ""),
                            "relation_type": rel.get("relation", ""),
                            "chunk_key": chunk_key
                        })
                
                batch_results.append((entities, relationships))
                
            except Exception as e:
                logger.warning(f"解析结果失败 {chunk_key}: {e}")
                batch_results.append(([], []))
        
        return batch_results

def create_parallel_extractor(llm_func, max_concurrent=4, batch_size=8):
    """创建并行实体提取器实例"""
    return ParallelEntityExtractor(llm_func, max_concurrent, batch_size)

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
        max_concurrent=4,
        batch_size=8
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
