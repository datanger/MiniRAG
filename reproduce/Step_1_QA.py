# from huggingface_hub import login
# your_token = "INPUT YOUR TOKEN HERE"
# login(your_token)

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载config.env配置文件
config_path = Path(__file__).parent.parent / "config.env"
load_dotenv(config_path)


import csv
from tqdm import trange
from minirag import MiniRAG, QueryParam
from minirag.llm import (
    hf_model_complete,
    hf_embed,
)
from minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

import argparse


def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG")
    parser.add_argument("--model", type=str, default="PHI")
    parser.add_argument("--outputpath", type=str, default="./logs/Default_output.csv")
    parser.add_argument("--workingdir", type=str, default="./LiHua-World")
    parser.add_argument("--datapath", type=str, default="./dataset/LiHua-World/data/")
    parser.add_argument(
        "--querypath", type=str, default="./dataset/LiHua-World/qa/query_set.csv"
    )
    # 并行优化参数 - 从config.env读取默认值
    parser.add_argument("--enable-parallel", action="store_true", 
                       help="启用并行实体提取")
    parser.add_argument("--max-concurrent", type=int, 
                       default=int(os.getenv("PARALLEL_ENTITY_EXTRACTION_MAX_CONCURRENT", 16)),
                       help="最大并发数")
    parser.add_argument("--batch-size", type=int, 
                       default=int(os.getenv("PARALLEL_ENTITY_EXTRACTION_BATCH_SIZE", 10)),
                       help="批处理大小")
    parser.add_argument("--enable-smart-batching", action="store_true",
                       help="启用智能批处理")
    args = parser.parse_args()
    return args


args = get_args()


# 从config.env读取LLM配置
LLM_BINDING = os.getenv("LLM_BINDING", "openai")
LLM_BINDING_HOST = os.getenv("LLM_BINDING_HOST", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# 如果命令行指定了模型，则覆盖config.env中的设置
if args.model != "PHI":  # PHI是默认值，如果指定其他模型则使用命令行参数
    if args.model == "GLM":
        LLM_MODEL = "THUDM/glm-edge-1.5b-chat"
    elif args.model == "MiniCPM":
        LLM_MODEL = "openbmb/MiniCPM3-4B"
    elif args.model == "qwen":
        LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
    elif args.model == "PHI":
        LLM_MODEL = "microsoft/Phi-3.5-mini-instruct"
    else:
        # 允许用户指定任意模型名称
        LLM_MODEL = args.model
        print(f"📝 使用用户指定的模型: {LLM_MODEL}")

print(f"🔧 LLM配置:")
print(f"   - 绑定类型: {LLM_BINDING}")
print(f"   - 服务器地址: {LLM_BINDING_HOST}")
print(f"   - 模型名称: {LLM_MODEL}")
print(f"   - 配置来源: {'命令行覆盖' if args.model != 'PHI' else 'config.env'}")

WORKING_DIR = args.workingdir
DATA_PATH = args.datapath
QUERY_PATH = args.querypath
OUTPUT_PATH = args.outputpath
print("=" * 60)
print("MiniRAG QA 脚本配置")
print("=" * 60)
print(f"LLM模型: {LLM_MODEL}")
print(f"工作目录: {WORKING_DIR}")
print(f"数据路径: {DATA_PATH}")
print(f"查询文件: {QUERY_PATH}")
print(f"输出文件: {OUTPUT_PATH}")
print(f"配置文件: {config_path}")
print("=" * 60)

# 显示从config.env读取的关键配置
print("📋 从config.env读取的配置:")
print(f"   - 最大Token数: {os.getenv('MAX_TOKENS', '200')}")
print(f"   - 嵌入维度: {os.getenv('EMBEDDING_DIM', '384')}")
print(f"   - 最大嵌入Token: {os.getenv('MAX_EMBED_TOKENS', '1000')}")
print(f"   - 最大并行插入: {os.getenv('MAX_PARALLEL_INSERT', '2')}")
print(f"   - 嵌入批处理大小: {os.getenv('EMBEDDING_BATCH_NUM', '32')}")
print(f"   - 嵌入函数最大异步数: {os.getenv('EMBEDDING_FUNC_MAX_ASYNC', '16')}")
print(f"   - LLM模型最大异步数: {os.getenv('LLM_MODEL_MAX_ASYNC', '16')}")
print("=" * 60)


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# 根据命令行参数和环境变量配置并行优化
parallel_config = {}

# 检查是否启用并行优化（命令行参数优先，其次环境变量）
enable_parallel = args.enable_parallel or os.getenv("ENABLE_PARALLEL_ENTITY_EXTRACTION", "true").lower() == "true"

if enable_parallel:
    parallel_config.update({
        "enable_parallel_entity_extraction": True,
        "parallel_entity_extraction_max_concurrent": args.max_concurrent,
        "parallel_entity_extraction_batch_size": args.batch_size,
        "enable_smart_batching": args.enable_smart_batching or os.getenv("ENABLE_SMART_BATCHING", "true").lower() == "true",
    })
    print(f"🚀 启用并行优化:")
    print(f"   - 最大并发数: {args.max_concurrent} (来自: {'命令行' if args.enable_parallel else 'config.env'})")
    print(f"   - 批处理大小: {args.batch_size} (来自: {'命令行' if args.enable_parallel else 'config.env'})")
    print(f"   - 智能批处理: {parallel_config['enable_smart_batching']}")
    print(f"   - 配置来源: config.env + 命令行覆盖")
else:
    print("⚡ 使用默认串行模式")

# 根据config.env中的绑定类型选择LLM函数
def get_llm_function():
    """根据配置选择LLM函数"""
    if LLM_BINDING == "openai":
        # 使用OpenAI兼容的API
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url=LLM_BINDING_HOST,
                api_key=os.getenv("LLM_BINDING_API_KEY")
            )
            
            async def openai_llm_func(prompt, **kwargs):
                try:
                    # 过滤掉MiniRAG特有的参数
                    filtered_kwargs = {}
                    for key, value in kwargs.items():
                        if key not in ['hashing_kv', 'keyword_extraction', 'system_prompt', 'history_messages']:
                            filtered_kwargs[key] = value
                    
                    response = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=int(os.getenv("MAX_TOKENS", 200)),
                        **filtered_kwargs
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"OpenAI API调用失败: {e}")
                    return f"API调用失败: {str(e)}"
            
            return openai_llm_func
            
        except ImportError:
            print("⚠️ OpenAI库未安装，回退到HuggingFace模型")
            return hf_model_complete
    
    elif LLM_BINDING == "ollama":
        # 使用Ollama本地模型
        try:
            from minirag.llm import ollama_complete
            
            async def ollama_llm_func(prompt, **kwargs):
                return await ollama_complete(prompt, model=LLM_MODEL, **kwargs)
            
            return ollama_llm_func
            
        except ImportError:
            print("⚠️ Ollama支持未安装，回退到HuggingFace模型")
            return hf_model_complete
    
    else:
        # 默认使用HuggingFace模型
        print(f"⚠️ 未知的LLM绑定类型: {LLM_BINDING}，使用HuggingFace模型")
        return hf_model_complete

# 获取LLM函数
llm_func = get_llm_function()

rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_func,
    llm_model_max_token_size=int(os.getenv("MAX_TOKENS", 200)),
    llm_model_name=LLM_MODEL,
    # 从config.env读取高级配置
    max_parallel_insert=int(os.getenv("MAX_PARALLEL_INSERT", 2)),
    embedding_batch_num=int(os.getenv("EMBEDDING_BATCH_NUM", 32)),
    embedding_func_max_async=int(os.getenv("EMBEDDING_FUNC_MAX_ASYNC", 16)),
    llm_model_max_async=int(os.getenv("LLM_MODEL_MAX_ASYNC", 16)),
    # 嵌入函数配置
    embedding_func=EmbeddingFunc(
        embedding_dim=int(os.getenv("EMBEDDING_DIM", 384)),
        max_token_size=int(os.getenv("MAX_EMBED_TOKENS", 1000)),
        func=lambda texts: hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
            embed_model=AutoModel.from_pretrained(EMBEDDING_MODEL),
        ),
    ),
    **parallel_config  # 应用并行配置
)

# Now QA
QUESTION_LIST = []
GA_LIST = []
with open(QUERY_PATH, mode="r", encoding="utf-8") as question_file:
    reader = csv.DictReader(question_file)
    for row in reader:
        QUESTION_LIST.append(row["Question"])
        GA_LIST.append(row["Gold Answer"])


def run_experiment(output_path):
    headers = ["Question", "Gold Answer", "minirag"]

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 创建输出目录: {output_dir}")

    q_already = []
    if os.path.exists(output_path):
        with open(output_path, mode="r", encoding="utf-8") as question_file:
            reader = csv.DictReader(question_file)
            for row in reader:
                q_already.append(row["Question"])

    row_count = len(q_already)
    print("row_count", row_count)

    with open(output_path, mode="a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        if row_count == 0:
            writer.writerow(headers)

        for QUESTIONid in trange(row_count, len(QUESTION_LIST)):  #
            QUESTION = QUESTION_LIST[QUESTIONid]
            Gold_Answer = GA_LIST[QUESTIONid]
            print()
            print("QUESTION", QUESTION)
            print("Gold_Answer", Gold_Answer)

            try:
                minirag_answer = (
                    rag.query(QUESTION, param=QueryParam(mode="mini"))
                    .replace("\n", "")
                    .replace("\r", "")
                )
            except Exception as e:
                print("Error in minirag_answer", e)
                minirag_answer = "Error"

            writer.writerow([QUESTION, Gold_Answer, minirag_answer])

    print(f"Experiment data has been recorded in the file: {output_path}")


if __name__ == "__main__":
    print("\n📖 使用说明:")
    print("基础用法:")
    print("  python Step_1_QA.py --model PHI")
    print("\n使用config.env中的模型:")
    print("  python Step_1_QA.py")
    print("\n指定任意模型名称:")
    print("  python Step_1_QA.py --model your-model-name")
    print("\n启用并行优化:")
    print("  python Step_1_QA.py --enable-parallel --max-concurrent 8")
    print("\n完整参数:")
    print("  python Step_1_QA.py --model your-model --enable-parallel --max-concurrent 8 --batch-size 10 --enable-smart-batching")
    print("\n📝 配置说明:")
    print("  - 脚本会自动读取 config.env 中的配置")
    print("  - 命令行参数会覆盖 config.env 中的设置")
    print("  - 并行优化默认从 config.env 启用")
    print("  - 支持任意模型名称，不再限制预定义选项")
    print("\n" + "=" * 60)
    
    run_experiment(OUTPUT_PATH)
