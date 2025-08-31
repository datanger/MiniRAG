# 🚀 MiniRAG 使用指南

## 📋 概述

这是一个完整的MiniRAG系统，包含：
- **RAG API客户端** - 调用MiniRAG服务器API
- **终端问答CLI** - 交互式问答界面
- **RAG系统构建器** - 批量构建RAG知识库
- **服务器启动脚本** - 一键启动MiniRAG服务器

## 🛠️ 安装依赖

```bash
pip install aiohttp requests python-dotenv
```

## 🚀 快速开始

### 1. 启动MiniRAG服务器

```bash
cd chatSDW
python start_rag_server.py
```

**✨ 完全环境变量配置**：启动脚本会自动读取项目根目录下的 `config.env` 文件，所有参数都通过环境变量设置，无需任何命令行参数！

服务器将根据配置文件自动设置：
- 🖥️ 服务器配置：主机地址、端口、SSL等
- 📁 目录配置：工作目录、输入目录
- 🤖 LLM配置：绑定类型、主机、API密钥、模型等
- 📊 嵌入配置：绑定类型、主机、API密钥、模型、维度等
- ⚙️ RAG配置：分块大小、异步操作、token限制等
- 🗄️ 存储配置：KV存储、向量存储、图存储、文档状态存储
- 🔒 安全配置：API密钥、HTTPS等
- 📈 高级配置：文件监控、实体提取、多模态等

### 2. 使用终端问答CLI

```bash
# 完全自动配置（推荐）
python rag_cli.py

# 或指定特定配置
python rag_cli.py --server http://localhost:9721 --api-key YOUR_API_KEY
```

支持的命令：
- `/help` - 显示帮助信息
- `/health` - 检查服务器状态
- `/docs` - 显示已索引文档
- `/scan` - 扫描并索引新文档
- `/upload <文件路径> [描述]` - 上传文档
- `/insert <文本内容> [描述]` - 插入文本
- `/clear` - 清空对话历史
- `/quit` - 退出程序

### 3. 使用RAG构建器

#### 从目录构建RAG系统
```bash
# 完全自动配置（推荐）
python rag_builder.py --directory ./dataset/kotei

# 或指定特定配置
python rag_builder.py --server http://localhost:9721 --api-key YOUR_API_KEY --directory ./dataset/kotei
```

#### 指定文件类型
```bash
python rag_builder.py --directory ./dataset/kotei --file-types .txt .pdf .docx
```

#### 从文本文件构建
```bash
python rag_builder.py --text-file texts.json
```

`texts.json` 格式：
```json
[
    {
        "content": "这是第一个文本内容...",
        "description": "第一个文本描述"
    },
    {
        "content": "这是第二个文本内容...",
        "description": "第二个文本描述"
    }
]
```

## 📖 详细使用说明

### RAG API客户端 (rag_client.py)

提供两种客户端：
- `MiniRAGClient` - 异步客户端
- `MiniRAGClientSync` - 同步客户端

```python
from rag_client import MiniRAGClient, MiniRAGClientSync

# 异步客户端
async with MiniRAGClient("http://localhost:9721") as client:
    # 健康检查
    health = await client.health_check()
    
    # 查询
    result = await client.query("你的问题", mode="mini")
    
    # 插入文本
    await client.insert_text("文本内容", "描述")
    
    # 上传文档
    await client.upload_document("文件路径", "描述")

# 同步客户端
client = MiniRAGClientSync("http://localhost:9721")
health = client.health_check()
result = client.query("你的问题", mode="mini")
```

### 终端问答CLI (rag_cli.py)

交互式问答界面，支持多轮对话：

```bash
python rag_cli.py --server http://localhost:9721 --api-key YOUR_API_KEY
```

特性：
- 🧠 智能问答（支持mini/light/naive三种模式）
- 💬 多轮对话支持
- 📁 文档管理功能
- 🔍 实时状态监控

### RAG系统构建器 (rag_builder.py)

批量构建RAG知识库：

```python
from rag_builder import RAGBuilder

async with RAGBuilder("http://localhost:9721") as builder:
    # 从目录构建
    result = await builder.build_from_directory("./dataset/kotei")
    
    # 从文本构建
    texts = [
        {"content": "内容1", "description": "描述1"},
        {"content": "内容2", "description": "描述2"}
    ]
    result = await builder.build_from_texts(texts)
    
    # 混合源构建
    result = await builder.build_from_mixed_sources(
        directories=["./docs", "./articles"],
        texts=texts,
        file_types=[".txt", ".pdf", ".docx"]
    )
```

## 🔧 配置选项

### 环境变量配置

所有工具都会自动读取项目根目录下的 `config.env` 文件：

```env
# 服务器配置
HOST=0.0.0.0
PORT=9721
LIGHTRAG_API_KEY=your_api_key_here

# LLM配置
LLM_BINDING=deepseek
LLM_BINDING_HOST=https://api.deepseek.com
LLM_BINDING_API_KEY=${DEEPSEEK_API_KEY}

# 存储配置
KV_STORAGE=JsonKVStorage
VECTOR_STORAGE=NanoVectorDBStorage
GRAPH_STORAGE=NetworkXStorage
DOC_STATUS_STORAGE=JsonDocStatusStorage
```

### 命令行参数

所有工具都支持以下参数（可选）：
- `--server` - 服务器地址（默认：从config.env自动读取）
- `--api-key` - API密钥（默认：从config.env自动读取）

**✨ 推荐使用方式**：直接运行工具，无需任何参数，所有配置自动从 `config.env` 读取！

**⚠️ 重要提醒**：所有工具都完全依赖 `config.env` 文件，请确保配置文件完整且正确！

## 📊 查询模式说明

### 1. Mini模式（推荐）
- 最智能的问答模式
- 使用LLM进行关键词提取
- 向量搜索实体和关系
- 图遍历获取上下文
- LLM生成最终答案

### 2. Light模式
- 轻量级查询模式
- 快速向量检索
- 适合简单问答

### 3. Naive模式
- 简单向量检索
- 直接返回相似文档
- 速度最快

## 🎯 使用场景

### 1. 文档问答
```bash
# 启动CLI
python rag_cli.py

# 在CLI中提问
💭 [mini] 请输入问题或命令: 请介绍一下VPN的使用方法
```

### 2. 批量文档处理
```bash
# 处理整个目录
python rag_builder.py --directory ./dataset/kotei

# 处理特定文件类型
python rag_builder.py --directory ./dataset/kotei --file-types .pdf .docx
```

### 3. 自定义文本插入
```bash
# 插入单条文本
python rag_cli.py
💭 [mini] 请输入问题或命令: /insert 这是一段重要的文本内容 重要说明文档
```

## 🚨 故障排除

### 常见问题

1. **服务器连接失败**
   - 检查MiniRAG服务器是否运行
   - 确认端口9721是否被占用
   - 检查防火墙设置

2. **API认证失败**
   - 确认API密钥是否正确
   - 检查环境变量设置
   - 验证服务器认证配置

3. **文档处理失败**
   - 检查文件格式是否支持
   - 确认文件编码（推荐UTF-8）
   - 查看服务器日志

### 日志查看

```bash
# 检查服务器状态
python rag_cli.py
💭 [mini] 请输入问题或命令: /health

# 查看已索引文档
💭 [mini] 请输入问题或命令: /docs
```

## 🔗 相关链接

- **API文档**: http://localhost:9721/docs
- **健康检查**: http://localhost:9721/health
- **项目主页**: https://github.com/your-repo/minirag

## 📞 技术支持

如果遇到问题，请：
1. 查看服务器日志
2. 检查配置文件
3. 确认依赖包版本
4. 提交Issue到项目仓库

---

**🎉 享受你的MiniRAG体验！**
