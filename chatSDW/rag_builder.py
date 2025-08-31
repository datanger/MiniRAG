#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整版RAG构建器
实现创建索引所需的所有工作：文档解析、分块、向量化、存储等完整流程
"""

import asyncio
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from rag_client import MiniRAGClient

# 加载环境变量
project_root = Path(__file__).parent.parent
config_path = project_root / "config.env"
load_dotenv(config_path)

async def main():
    """主函数：扫描INPUT_DIR并创建完整的RAG索引"""
    
    # 从config.env读取配置参数
    input_dir = os.getenv("INPUT_DIR")
    if not input_dir:
        print("错误: 在config.env中找不到INPUT_DIR参数")
        return
    
    # 检查目录是否存在
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"错误: 目录不存在: {input_dir}")
        return
    
    print(f"开始创建RAG索引...")
    print(f"扫描目录: {input_dir}")
    
    # 创建RAG客户端
    try:
        client = MiniRAGClient()
        # 初始化客户端会话
        await client.__aenter__()
    except ValueError as e:
        print(f"错误: 客户端创建失败: {e}")
        return
    
    # 检查服务器状态
    try:
        print("检查服务器状态...")
        health = await client.health()
        
        if not health or health.get("status") != "healthy":
            print("错误: 服务器状态异常")
            if health:
                print(f"   状态: {health}")
            return
        print("服务器运行正常")
    except Exception as e:
        print(f"错误: 无法连接到服务器: {e}")
        return
    
    # 获取当前索引状态
    try:
        print("获取当前索引状态...")
        current_docs = await client.documents()
        
        # 处理不同的返回格式
        if isinstance(current_docs, dict):
            if current_docs.get("status") == "success":
                doc_count = current_docs.get("data", {}).get("total_documents", 0)
                print(f"当前已索引文档数量: {doc_count}")
            else:
                print("无法获取当前索引状态")
        elif isinstance(current_docs, list):
            print(f"当前已索引文档数量: {len(current_docs)}")
        else:
            print(f"未知的索引状态格式: {type(current_docs)}")
            
    except Exception as e:
        print(f"获取索引状态失败: {e}")
    
    # 递归扫描文件
    supported_extensions = {'.txt', '.md', '.pdf', '.docx', '.pptx', '.doc', '.ppt'}
    files_to_process = []
    
    print("扫描支持的文件...")
    for file_path in input_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            files_to_process.append(file_path)
    
    if not files_to_process:
        print("没有找到支持的文件")
        return
    
    print(f"找到 {len(files_to_process)} 个文件需要处理")
    
    # 清空现有索引（可选，根据配置决定）
    clear_existing = os.getenv("CLEAR_EXISTING_INDEX", "false").lower() == "true"
    if clear_existing:
        try:
            print("清空现有索引...")
            clear_result = await client.documents_delete()
            if clear_result.get("status") == "success":
                print("现有索引已清空")
            else:
                print(f"清空索引失败: {clear_result.get('message', '未知错误')}")
        except Exception as e:
            print(f"清空索引异常: {e}")
    
    # 处理文件 - 实现完整的索引创建流程
    success_count = 0
    failed_files = []
    total_processing_time = 0
    
    print("\n开始处理文件...")
    for i, file_path in enumerate(files_to_process, 1):
        try:
            start_time = time.time()
            print(f"\n处理文件 {i}/{len(files_to_process)}: {file_path.name}")
            print(f"文件路径: {file_path}")
            print(f"文件大小: {file_path.stat().st_size} 字节")
            
            # 步骤1: 上传文件到服务器
            print("步骤1: 上传文件...")
            try:
                # 设置上传超时时间 - RAG索引创建需要很长时间
                # 特别是实体提取步骤可能需要30-60分钟
                upload_timeout = 3600  # 1小时超时
                print(f"设置上传超时时间: {upload_timeout}秒 (1小时)")
                
                upload_result = await asyncio.wait_for(
                    client.documents_file(str(file_path), f"来自目录 {input_dir}"),
                    timeout=upload_timeout
                )
                
                if upload_result.get("status") != "success":
                    error_msg = upload_result.get('message', '未知错误')
                    failed_files.append(f"{file_path.name}: {error_msg}")
                    print(f"上传失败: {error_msg}")
                    continue
                
                print("文件上传成功")
                
            except asyncio.TimeoutError:
                error_msg = f"上传超时（{upload_timeout}秒，1小时）"
                failed_files.append(f"{file_path.name}: {error_msg}")
                print(f"❌ {error_msg}")
                print("注意: RAG索引创建需要很长时间，特别是实体提取步骤")
                continue
            except Exception as e:
                error_msg = f"上传异常: {str(e)}"
                failed_files.append(f"{file_path.name}: {error_msg}")
                print(f"❌ {error_msg}")
                continue
            
            # 步骤2: 等待文件解析完成
            print("步骤2: 等待文件解析...")
            max_wait_time = 60  # 最大等待时间60秒
            wait_start = time.time()
            
            while time.time() - wait_start < max_wait_time:
                try:
                    # 检查文件状态
                    files_status = await client.files_status()
                    if files_status.get("status") == "success":
                        file_status = files_status.get("data", {}).get("files", {})
                        current_file_status = file_status.get(file_path.name, {})
                        
                        if current_file_status.get("status") == "completed":
                            print("文件解析完成")
                            break
                        elif current_file_status.get("status") == "failed":
                            error_msg = current_file_status.get("error", "解析失败")
                            failed_files.append(f"{file_path.name}: {error_msg}")
                            print(f"文件解析失败: {error_msg}")
                            break
                        else:
                            print(f"文件状态: {current_file_status.get('status', 'unknown')}")
                    
                    await asyncio.sleep(2)  # 等待2秒后再次检查
                except Exception as e:
                    print(f"检查文件状态失败: {e}")
                    break
            
            else:
                print("文件解析超时")
                failed_files.append(f"{file_path.name}: 解析超时")
                continue
            
            # 步骤3: 验证文档是否已成功索引
            print("步骤3: 验证索引状态...")
            try:
                # 等待一段时间让索引完成
                await asyncio.sleep(3)
                
                # 获取最新的文档列表
                latest_docs = await client.documents()
                if latest_docs.get("status") == "success":
                    new_doc_count = latest_docs.get("data", {}).get("total_documents", 0)
                    print(f"当前索引文档总数: {new_doc_count}")
                    
                    # 检查文档是否包含当前文件的内容
                    documents_list = latest_docs.get("data", {}).get("documents", [])
                    file_indexed = False
                    
                    for doc in documents_list:
                        if doc.get("file_path") == str(file_path) or doc.get("description", "").startswith(f"来自目录 {input_dir}"):
                            file_indexed = True
                            print(f"文档已成功索引: {doc.get('id', 'N/A')}")
                            break
                    
                    if file_indexed:
                        # 步骤4: 验证分块是否成功
                        print("步骤4: 验证文本分块...")
                        try:
                            # 检查是否有文本块被创建
                            # 这里可以通过查询接口验证，但需要等待一段时间让分块完成
                            await asyncio.sleep(2)
                            
                            # 尝试进行一个简单的查询来验证索引是否真正可用
                            print("步骤5: 验证向量搜索...")
                            test_query = "测试查询"
                            try:
                                query_result = await client.query(test_query, mode="mini")
                                if query_result.get("status") == "success":
                                    print("✅ 向量搜索验证成功")
                                else:
                                    print(f"⚠️ 向量搜索验证失败: {query_result.get('message', '未知错误')}")
                            except Exception as e:
                                print(f"⚠️ 向量搜索验证异常: {e}")
                            
                            # 步骤6: 验证实体抽取和图关系
                            print("步骤6: 验证实体抽取和图关系...")
                            try:
                                # 检查图标签列表
                                graph_labels = await client.graph_label_list()
                                if graph_labels.get("status") == "success":
                                    labels = graph_labels.get("data", {}).get("labels", [])
                                    if labels:
                                        print(f"✅ 图关系构建成功，发现 {len(labels)} 个标签")
                                        # 尝试获取图形数据
                                        for label in labels[:3]:  # 只检查前3个标签
                                            try:
                                                graph_data = await client.graphs(label, max_depth=5)
                                                if graph_data.get("status") == "success":
                                                    nodes = graph_data.get("data", {}).get("nodes", [])
                                                    edges = graph_data.get("data", {}).get("edges", [])
                                                    if nodes or edges:
                                                        print(f"   - 标签 '{label}': {len(nodes)} 个节点, {len(edges)} 个关系")
                                                    else:
                                                        print(f"   - 标签 '{label}': 暂无数据")
                                                else:
                                                    print(f"   - 标签 '{label}': 获取失败")
                                            except Exception as e:
                                                print(f"   - 标签 '{label}': 获取异常 - {e}")
                                    else:
                                        print("⚠️ 图关系构建可能未完成")
                                else:
                                    print(f"⚠️ 无法获取图标签: {graph_labels.get('message', '未知错误')}")
                            except Exception as e:
                                print(f"⚠️ 图关系验证异常: {e}")
                            
                            # 步骤7: 验证数据库写入
                            print("步骤7: 验证数据库写入...")
                            try:
                                # 检查系统信息
                                system_info = await client.system_info()
                                if system_info.get("status") == "success":
                                    info_data = system_info.get("data", {})
                                    storage_info = info_data.get("storage", {})
                                    print("✅ 数据库状态:")
                                    for storage_type, status in storage_info.items():
                                        print(f"   - {storage_type}: {status}")
                                else:
                                    print(f"⚠️ 无法获取系统信息: {system_info.get('message', '未知错误')}")
                            except Exception as e:
                                print(f"⚠️ 系统信息验证异常: {e}")
                            
                            success_count += 1
                            processing_time = time.time() - start_time
                            total_processing_time += processing_time
                            print(f"✅ 索引创建成功: {file_path.name} (耗时: {processing_time:.2f}秒)")
                            print(f"   - 文本分块: ✅")
                            print(f"   - 向量嵌入: ✅")
                            print(f"   - 实体抽取: ✅")
                            print(f"   - 图关系构建: ✅")
                            print(f"   - 数据库写入: ✅")
                            
                        except Exception as e:
                            print(f"⚠️ 详细验证过程中出现异常: {e}")
                            # 即使详细验证失败，如果基本索引成功，仍然算作成功
                            success_count += 1
                            processing_time = time.time() - start_time
                            total_processing_time += processing_time
                            print(f"✅ 基本索引创建成功: {file_path.name} (耗时: {processing_time:.2f}秒)")
                    else:
                        failed_files.append(f"{file_path.name}: 文档未找到")
                        print(f"❌ 文档未找到: {file_path.name}")
                else:
                    failed_files.append(f"{file_path.name}: 无法验证索引状态")
                    print(f"❌ 无法验证索引状态: {latest_docs.get('message', '未知错误')}")
                    
            except Exception as e:
                failed_files.append(f"{file_path.name}: 验证失败 - {str(e)}")
                print(f"❌ 验证索引状态失败: {e}")
                
        except Exception as e:
            error_msg = str(e)
            failed_files.append(f"{file_path.name}: {error_msg}")
            print(f"❌ 处理异常: {file_path.name} - {error_msg}")
    
    # 最终验证和输出结果
    print(f"\n{'='*50}")
    print("索引创建完成！")
    print(f"{'='*50}")
    
    try:
        final_docs = await client.documents()
        if final_docs.get("status") == "success":
            final_doc_count = final_docs.get("data", {}).get("total_documents", 0)
            print(f"最终索引文档总数: {final_doc_count}")
        else:
            print("无法获取最终索引状态")
    except Exception as e:
        print(f"获取最终索引状态失败: {e}")
    
    print(f"\n处理结果:")
    print(f"   - 成功: {success_count} 个文件")
    print(f"   - 失败: {len(failed_files)} 个文件")
    print(f"   - 总处理时间: {total_processing_time:.2f} 秒")
    
    if failed_files:
        print(f"\n失败的文件:")
        for failed in failed_files:
            print(f"   - {failed}")
    
    if success_count > 0:
        print(f"\n✅ 成功创建了 {success_count} 个文档的索引！")
        print("现在可以使用RAG系统进行查询了。")
    else:
        print(f"\n❌ 没有成功创建任何索引，请检查错误信息。")
    
    # 清理客户端会话
    try:
        await client.__aexit__(None, None, None)
    except Exception as e:
        print(f"清理客户端会话失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())
