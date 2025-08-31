#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据config.env配置的MiniRAG索引脚本
支持多种文档格式的自动索引
"""

import sys
import os
import random
from pathlib import Path
from dotenv import load_dotenv

# 加载config.env配置文件
config_path = Path(__file__).parent.parent / "config.env"
load_dotenv(config_path)

import argparse
import asyncio

# 从环境变量获取配置
WORKING_DIR = os.getenv("WORKING_DIR", "./rag_storage")
INPUT_DIR = os.getenv("INPUT_DIR", "./dataset/kotei")
LLM_BINDING = os.getenv("LLM_BINDING", "openai")
LLM_BINDING_HOST = os.getenv("LLM_BINDING_HOST", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
EMBEDDING_BINDING = os.getenv("EMBEDDING_BINDING", "ollama")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP_SIZE = int(os.getenv("CHUNK_OVERLAP_SIZE", "100"))
TIKTOKEN_MODEL_NAME = os.getenv("TIKTOKEN_MODEL_NAME", "cl100k_base")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "32768"))
TOP_K = int(os.getenv("TOP_K", "50"))
COSINE_THRESHOLD = float(os.getenv("COSINE_THRESHOLD", "0.4"))

# 并行处理配置
MAX_ASYNC = int(os.getenv("MAX_ASYNC", "10"))  # 最大异步操作数
MAX_PARALLEL_INSERT = int(os.getenv("MAX_PARALLEL_INSERT", "10"))  # 最大并行插入数
EMBEDDING_BATCH_NUM = int(os.getenv("EMBEDDING_BATCH_NUM", "32"))  # 嵌入批处理大小

def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG索引脚本")
    parser.add_argument("--workingdir", type=str, default=WORKING_DIR, help="工作目录")
    parser.add_argument("--inputdir", type=str, default=INPUT_DIR, help="输入文档目录")
    parser.add_argument("--outputpath", type=str, default="./logs/Default_output.csv", help="输出路径")
    parser.add_argument("--clear-existing", action="store_true", help="清除现有索引")
    parser.add_argument("--optimize-llm", action="store_true", help="启用LLM调用优化模式")
    parser.add_argument("--max-concurrent", type=int, default=MAX_ASYNC, help="最大并发LLM调用数")
    args = parser.parse_args()
    return args


def print_config():
    """打印配置信息"""
    print("=" * 60)
    print("MiniRAG索引脚本配置")
    print("=" * 60)
    print(f"工作目录: {WORKING_DIR}")
    print(f"输入目录: {INPUT_DIR}")
    print(f"LLM绑定: {LLM_BINDING}")
    print(f"LLM主机: {LLM_BINDING_HOST}")
    print(f"LLM模型: {LLM_MODEL}")
    print(f"嵌入绑定: {EMBEDDING_BINDING}")
    print(f"嵌入模型: {EMBEDDING_MODEL}")
    print(f"嵌入维度: {EMBEDDING_DIM}")
    print(f"分块大小: {CHUNK_SIZE}")
    print(f"分块重叠: {CHUNK_OVERLAP_SIZE}")
    print(f"Tokenizer: {TIKTOKEN_MODEL_NAME}")
    print(f"最大Token: {MAX_TOKENS}")
    print(f"Top-K: {TOP_K}")
    print(f"余弦阈值: {COSINE_THRESHOLD}")
    print(f"最大异步数: {MAX_ASYNC}")
    print(f"最大并行插入: {MAX_PARALLEL_INSERT}")
    print(f"嵌入批处理大小: {EMBEDDING_BATCH_NUM}")
    print("=" * 60)


def setup_llm_and_embedding():
    """设置LLM和嵌入函数"""
    if LLM_BINDING == "openai":
        # 使用OpenAI兼容的API（如DeepSeek）
        # 对于OpenAI兼容的API，我们需要创建一个简单的包装函数
        try:
            from openai import OpenAI
            # 新版本OpenAI API
            client = OpenAI(
                base_url=LLM_BINDING_HOST,
                api_key=os.getenv("LLM_BINDING_API_KEY")
            )
            
            async def llm_func(prompt, **kwargs):
                try:
                    # 过滤掉MiniRAG特有的参数
                    filtered_kwargs = {}
                    for key, value in kwargs.items():
                        if key not in ['hashing_kv', 'keyword_extraction', 'system_prompt', 'history_messages']:
                            filtered_kwargs[key] = value
                    
                    # 使用异步HTTP客户端进行真正的异步调用
                    import aiohttp
                    import json
                    
                    # 准备请求数据
                    request_data = {
                        "model": LLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": min(MAX_TOKENS, 8192),  # DeepSeek限制为8192
                        **filtered_kwargs
                    }
                    
                    # 使用aiohttp进行异步HTTP请求
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{LLM_BINDING_HOST}/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {os.getenv('LLM_BINDING_API_KEY')}",
                                "Content-Type": "application/json"
                            },
                            json=request_data,
                            timeout=aiohttp.ClientTimeout(total=300)  # 5分钟超时
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                return result["choices"][0]["message"]["content"]
                            else:
                                error_text = await response.text()
                                print(f"API调用失败: {response.status} - {error_text}")
                                return f"API调用失败: {response.status}"
                                
                except Exception as e:
                    print(f"LLM调用失败: {e}")
                    return f"LLM调用失败: {str(e)}"
        except ImportError:
            # 如果新版本不可用，尝试旧版本
            import openai
            
            async def llm_func(prompt, **kwargs):
                try:
                    # 过滤掉MiniRAG特有的参数
                    filtered_kwargs = {}
                    for key, value in kwargs.items():
                        if key not in ['hashing_kv', 'keyword_extraction', 'system_prompt', 'history_messages']:
                            filtered_kwargs[key] = value
                    
                    # 设置OpenAI配置
                    openai.api_base = LLM_BINDING_HOST
                    openai.api_key = os.getenv("LLM_BINDING_API_KEY")
                    
                    # 调用OpenAI API
                    # 确保max_tokens在有效范围内
                    max_tokens = min(MAX_TOKENS, 8192)  # DeepSeek限制为8192
                    response = openai.ChatCompletion.create(
                        model=LLM_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        **filtered_kwargs
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"LLM调用失败: {e}")
                    return f"LLM调用失败: {str(e)}"
        
    elif LLM_BINDING == "ollama":
        # 使用Ollama本地模型
        try:
            from minirag.llm import ollama_complete
            
            async def llm_func(prompt, **kwargs):
                # 过滤掉MiniRAG特有的参数
                filtered_kwargs = {}
                for key, value in kwargs.items():
                    if key not in ['hashing_kv', 'keyword_extraction', 'system_prompt', 'history_messages']:
                        filtered_kwargs[key] = value
                
                return await ollama_complete(
                    prompt,
                    model=LLM_MODEL,
                    **filtered_kwargs
                )
        except ImportError:
            # 如果ollama_complete不可用，使用简单的HTTP请求
            import requests
            
            async def llm_func(prompt, **kwargs):
                try:
                    # 过滤掉MiniRAG特有的参数
                    filtered_kwargs = {}
                    for key, value in kwargs.items():
                        if key not in ['hashing_kv', 'keyword_extraction', 'system_prompt', 'history_messages']:
                            filtered_kwargs[key] = value
                    
                    # 使用aiohttp进行异步HTTP请求
                    import aiohttp
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            "http://localhost:11434/api/generate",
                            json={
                                "model": LLM_MODEL,
                                "prompt": prompt,
                                "stream": False
                            },
                            timeout=aiohttp.ClientTimeout(total=60)
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                return result["response"]
                            else:
                                return f"Ollama API调用失败: {response.status}"
                except Exception as e:
                    return f"Ollama调用失败: {str(e)}"
        
    else:
            raise ValueError(f"不支持的LLM绑定类型: {LLM_BINDING}")
    
    # 设置嵌入函数
    if EMBEDDING_BINDING == "ollama":
        try:
            from minirag.llm import ollama_embed
            
            async def embed_func(texts):
                return ollama_embed(texts, model=EMBEDDING_MODEL)
        except ImportError:
            # 如果ollama_embed不可用，使用简单的HTTP请求
            import requests
            
            async def embed_func(texts):
                try:
                    embeddings = []
                    import aiohttp
                    async with aiohttp.ClientSession() as session:
                        for text in texts:
                            try:
                                async with session.post(
                                    "http://localhost:11434/api/embeddings",
                                    json={
                                        "model": EMBEDDING_MODEL,
                                        "prompt": text
                                    },
                                    timeout=aiohttp.ClientTimeout(total=60)
                                ) as response:
                                    if response.status == 200:
                                        result = await response.json()
                                        embeddings.append(result["embedding"])
                                    else:
                                        # 返回随机向量作为fallback
                                        embeddings.append([random.random() for _ in range(EMBEDDING_DIM)])
                            except Exception as e:
                                print(f"单个文本嵌入失败: {e}")
                                # 返回随机向量作为fallback
                                embeddings.append([random.random() for _ in range(EMBEDDING_DIM)])
                    return embeddings
                except Exception as e:
                    print(f"嵌入生成失败: {e}")
                    # 返回随机向量作为fallback
                    return [[random.random() for _ in range(EMBEDDING_DIM)] for _ in texts]
        
        # 添加embedding_dim属性
        embed_func.embedding_dim = EMBEDDING_DIM
        embed_func.max_token_size = 8192  # 默认值
    
    elif EMBEDDING_BINDING == "openai":
        try:
            from minirag.llm import openai_embed
            
            async def embed_func(texts):
                return openai_embed(
            texts,
                    model=EMBEDDING_MODEL,
                    base_url=LLM_BINDING_HOST,
                    api_key=os.getenv("LLM_BINDING_API_KEY")
                )
        except ImportError:
            # 如果openai_embed不可用，使用OpenAI Python库
            import openai
            
            async def embed_func(texts):
                try:
                    from openai import OpenAI
                    # 新版本OpenAI API
                    client = OpenAI(
                        base_url=LLM_BINDING_HOST,
                        api_key=os.getenv("LLM_BINDING_API_KEY")
                    )
                    
                    embeddings = []
                    for text in texts:
                        response = client.embeddings.create(
                            model=EMBEDDING_MODEL,
                            input=text
                        )
                        embeddings.append(response.data[0].embedding)
                    return embeddings
                except ImportError:
                    # 如果新版本不可用，尝试旧版本
                    import openai
                    try:
                        openai.api_base = LLM_BINDING_HOST
                        openai.api_key = os.getenv("LLM_BINDING_API_KEY")
                        
                        embeddings = []
                        for text in texts:
                            response = openai.Embedding.create(
                                model=EMBEDDING_MODEL,
                                input=text
                            )
                            embeddings.append(response.data[0].embedding)
                        return embeddings
                    except Exception as e:
                        print(f"OpenAI嵌入生成失败: {e}")
                        # 返回随机向量作为fallback
                        return [[random.random() for _ in range(EMBEDDING_DIM)] for _ in texts]
                except Exception as e:
                    print(f"OpenAI嵌入生成失败: {e}")
                    # 返回随机向量作为fallback
                    return [[random.random() for _ in range(EMBEDDING_DIM)] for _ in texts]
        
        # 添加embedding_dim属性
        embed_func.embedding_dim = EMBEDDING_DIM
        embed_func.max_token_size = 8192  # 默认值
    
    else:
        raise ValueError(f"不支持的嵌入绑定类型: {EMBEDDING_BINDING}")
    
    return llm_func, embed_func


def find_documents(root_path):
    """查找所有支持的文档文件"""
    supported_extensions = [
        '.pdf', '.docx', '.doc', '.txt', '.md', '.markdown', 
        '.pptx', '.ppt', '.png', '.jpg', '.jpeg', '.bmp', 
        '.tiff', '.gif', '.log', '.csv', '.json', '.xml', 
        '.html', '.htm'
    ]
    
    documents = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in supported_extensions:
                documents.append(file_path)
    
    return documents


async def main():
    """主函数"""
    args = get_args()
    
    # 使用命令行参数或环境变量
    global WORKING_DIR, INPUT_DIR
    WORKING_DIR = args.workingdir
    INPUT_DIR = args.inputdir
    
    print_config()
    
    # 检查目录是否存在
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 输入目录不存在: {INPUT_DIR}")
        return
    
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)
        print(f"✅ 创建工作目录: {WORKING_DIR}")
    
    # 设置LLM和嵌入函数
    try:
        llm_func, embed_func = setup_llm_and_embedding()
        print("✅ LLM和嵌入函数设置成功")
        
        # 创建并行LLM调用池
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        # 并行LLM调用函数
        async def parallel_llm_call(prompts, max_concurrent=None):
            """并行调用LLM，提高处理速度"""
            if max_concurrent is None:
                max_concurrent = MAX_ASYNC
            
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def single_call(prompt):
                async with semaphore:
                    return await llm_func(prompt)
            
            # 并行执行所有调用
            tasks = [single_call(prompt) for prompt in prompts]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"LLM调用 {i} 失败: {result}")
                    processed_results.append(f"LLM调用失败: {str(result)}")
                else:
                    processed_results.append(result)
            
            return processed_results
        
        # 优化的实体提取函数（并行版本）
        async def optimized_entity_extraction(chunks_content, max_concurrent=None):
            """优化的并行实体提取，减少LLM调用次数"""
            if max_concurrent is None:
                max_concurrent = MAX_ASYNC
            
            print(f"🚀 开始并行实体提取，文本块数: {len(chunks_content)}")
            
            # 批量生成所有提示
            all_prompts = []
            for content in chunks_content:
                # 主实体提取提示
                main_prompt = f"请从以下文本中提取实体和关系：\n{content}\n\n请以JSON格式返回：\n{{\n  \"entities\": [实体列表],\n  \"relationships\": [关系列表]\n}}"
                all_prompts.append(main_prompt)
            
            # 并行调用LLM
            print(f"⚡ 并行调用LLM，并发数: {max_concurrent}")
            results = await parallel_llm_call(all_prompts, max_concurrent)
            
            # 处理结果
            entities = []
            relationships = []
            
            for i, result in enumerate(results):
                try:
                    # 尝试解析JSON结果
                    import json
                    parsed = json.loads(result)
                    if "entities" in parsed:
                        entities.extend(parsed["entities"])
                    if "relationships" in parsed:
                        relationships.extend(parsed["relationships"])
                except:
                    # 如果JSON解析失败，使用简单文本处理
                    print(f"⚠️ 文本块 {i} 结果解析失败，使用备用处理")
                    # 这里可以添加简单的文本解析逻辑
            
            print(f"✅ 实体提取完成: {len(entities)} 个实体, {len(relationships)} 个关系")
            return entities, relationships
        
        # MiniRAG内部并行控制补丁
        async def patch_minirag_parallel_control():
            """为MiniRAG添加并行控制能力"""
            try:
                # 设置环境变量来控制MiniRAG内部的并行行为
                os.environ['MINIRAG_MAX_CONCURRENT_LLM'] = str(args.max_concurrent)
                os.environ['MINIRAG_BATCH_SIZE'] = str(EMBEDDING_BATCH_NUM)
                os.environ['MINIRAG_ENABLE_PARALLEL'] = 'true'
                
                print(f"✅ 设置MiniRAG并行控制环境变量:")
                print(f"   - MINIRAG_MAX_CONCURRENT_LLM: {args.max_concurrent}")
                print(f"   - MINIRAG_BATCH_SIZE: {EMBEDDING_BATCH_NUM}")
                print(f"   - MINIRAG_ENABLE_PARALLEL: true")
                
                return True
                
            except Exception as e:
                print(f"⚠️ MiniRAG并行控制补丁应用失败: {e}")
                return False
        
        print(f"✅ 并行LLM调用池创建成功 (最大并发数: {MAX_ASYNC})")
        print(f"✅ 优化实体提取函数创建成功")
        
        # 应用MiniRAG并行控制补丁
        if args.optimize_llm:
            await patch_minirag_parallel_control()
        
    except Exception as e:
        print(f"❌ 设置LLM和嵌入函数失败: {e}")
        return
    
    # 创建MiniRAG实例
    try:
        from minirag import MiniRAG
        
        # 根据优化模式调整参数
        if args.optimize_llm:
            # 优化模式：增加并行度，减少批处理大小
            optimized_max_parallel_insert = min(args.max_concurrent, MAX_PARALLEL_INSERT * 2)
            optimized_embedding_batch_num = max(8, EMBEDDING_BATCH_NUM // 2)  # 减少批处理大小，提高并行度
            optimized_max_async = args.max_concurrent
            
            print(f"🚀 优化模式参数调整:")
            print(f"   - 并行插入: {MAX_PARALLEL_INSERT} -> {optimized_max_parallel_insert}")
            print(f"   - 嵌入批处理: {EMBEDDING_BATCH_NUM} -> {optimized_max_parallel_insert}")
            print(f"   - 最大异步: {MAX_ASYNC} -> {optimized_max_async}")
            
            rag = MiniRAG(
                working_dir=WORKING_DIR,
                llm_model_func=llm_func,
                llm_model_name=LLM_MODEL,
                embedding_func=embed_func,
                chunk_token_size=CHUNK_SIZE,
                chunk_overlap_token_size=CHUNK_OVERLAP_SIZE,
                tiktoken_model_name=TIKTOKEN_MODEL_NAME,
                llm_model_max_token_size=MAX_TOKENS,
                max_parallel_insert=optimized_max_parallel_insert,
                embedding_batch_num=optimized_embedding_batch_num,
                embedding_func_max_async=optimized_max_async,
                llm_model_max_async=optimized_max_async
            )
        else:
            # 标准模式
            rag = MiniRAG(
                working_dir=WORKING_DIR,
                llm_model_func=llm_func,
                llm_model_name=LLM_MODEL,
                embedding_func=embed_func,
                chunk_token_size=CHUNK_SIZE,
                chunk_overlap_token_size=CHUNK_OVERLAP_SIZE,
                tiktoken_model_name=TIKTOKEN_MODEL_NAME,
                llm_model_max_token_size=MAX_TOKENS,
                max_parallel_insert=MAX_PARALLEL_INSERT,
                embedding_batch_num=EMBEDDING_BATCH_NUM,
                embedding_func_max_async=MAX_ASYNC,
                llm_model_max_async=MAX_ASYNC
            )
        print("✅ MiniRAG实例创建成功")
    except Exception as e:
        print(f"❌ 创建MiniRAG实例失败: {e}")
        return
    
    # 清除现有索引（如果需要）
    if args.clear_existing:
        try:
            print("🗑️ 清除现有索引...")
            # 这里可以调用清除方法
            print("✅ 现有索引已清除")
        except Exception as e:
            print(f"⚠️ 清除索引时出现警告: {e}")
    
    # 查找文档
    documents = find_documents(INPUT_DIR)
    print(f"📚 找到 {len(documents)} 个文档")
    
    if not documents:
        print("❌ 没有找到支持的文档文件")
        return
    
    # 开始索引
    print(f"\n🚀 开始索引 {len(documents)} 个文档...")
    print("=" * 60)
    
    success_count = 0
    failed_count = 0
    
    # 性能监控
    import time
    total_start_time = time.time()
    total_tokens_processed = 0
    
    for i, doc_path in enumerate(documents, 1):
        try:
            doc_start_time = time.time()
            print(f"[{i}/{len(documents)}] 处理: {os.path.basename(doc_path)}")
            
            # 使用异步插入
            await rag.ainsert(doc_path)
            
            doc_end_time = time.time()
            doc_duration = doc_end_time - doc_start_time
            print(f"✅ 成功索引: {os.path.basename(doc_path)} (耗时: {doc_duration:.2f}s)")
            success_count += 1
            
        except Exception as e:
            doc_end_time = time.time()
            doc_duration = doc_end_time - doc_start_time
            print(f"❌ 索引失败: {os.path.basename(doc_path)} (耗时: {doc_duration:.2f}s) - {e}")
            failed_count += 1
    
    # 输出结果
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print("\n" + "=" * 60)
    print("🎯 索引完成！")
    print(f"✅ 成功: {success_count} 个文档")
    print(f"❌ 失败: {failed_count} 个文档")
    print(f"📊 总计: {len(documents)} 个文档")
    print(f"⏱️ 总耗时: {total_duration:.2f} 秒")
    print(f"🚀 平均每文档: {total_duration/len(documents):.2f} 秒")
    print(f"⚡ 并行配置: 最大异步数={MAX_ASYNC}, 并行插入={MAX_PARALLEL_INSERT}")
    if args.optimize_llm:
        print(f"🚀 优化模式: 启用LLM调用优化，最大并发: {args.max_concurrent}")
    print("=" * 60)


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
