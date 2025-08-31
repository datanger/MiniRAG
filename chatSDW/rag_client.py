#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniRAG API 客户端
用于调用 minirag_server.py 的API接口实现RAG功能
"""

import os
import json
import asyncio
import aiohttp
import requests
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv("config.env")

class MiniRAGClient:
    """MiniRAG API 客户端类"""
    
    def __init__(self, base_url: str = None, api_key: Optional[str] = None):
        # 完全从config.env读取配置
        if base_url is None:
            host = os.getenv("HOST")
            port = os.getenv("PORT")
            if not host or not port:
                raise ValueError("HOST和PORT必须在config.env中配置")
            base_url = f"http://{host}:{port}"
        
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or os.getenv("LIGHTRAG_API_KEY")
        self.session = None
        self.logger = logging.getLogger(__name__)
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers
    
    async def health(self) -> Dict[str, Any]:
        """检查服务器健康状态"""
        try:
            if not self.session:
                return {"status": "error", "message": "会话未初始化"}
            
            async with self.session.get(f"{self.base_url}/health", headers=self._get_headers()) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
        except Exception as e:
            self.logger.error(f"健康检查失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def query(self, question: str, mode: str = "mini") -> Dict[str, Any]:
        """查询RAG系统"""
        try:
            url = f"{self.base_url}/query"
            payload = {
                "query": question,
                "mode": mode,
                "stream": False,
                "only_need_context": False
            }
            
            async with self.session.post(url, json=payload, headers=self._get_headers()) as response:
                return await response.json()
        except Exception as e:
            self.logger.error(f"查询失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def query_stream(self, question: str, mode: str = "mini") -> Dict[str, Any]:
        """流式查询RAG系统"""
        try:
            url = f"{self.base_url}/query/stream"
            payload = {
                "query": question,
                "mode": mode,
                "stream": True,
                "only_need_context": False
            }
            
            async with self.session.post(url, json=payload, headers=self._get_headers()) as response:
                return await response.json()
        except Exception as e:
            self.logger.error(f"流式查询失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def documents_text(self, text: str, description: str = "") -> Dict[str, Any]:
        """插入文本到RAG系统"""
        try:
            url = f"{self.base_url}/documents/text"
            payload = {"text": text, "description": description}
            
            async with self.session.post(url, json=payload, headers=self._get_headers()) as response:
                return await response.json()
        except Exception as e:
            self.logger.error(f"文本插入失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def documents_file(self, file_path: Union[str, Path], description: str = "") -> Dict[str, Any]:
        """插入文件到RAG系统"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return {"status": "error", "message": f"文件不存在: {file_path}"}
            
            url = f"{self.base_url}/documents/file"
            
            # 使用异步方式读取文件
            file_content = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: open(file_path, 'rb').read()
            )
            
            # 准备multipart数据
            data = aiohttp.FormData()
            data.add_field('file', file_content, filename=file_path.name, content_type='application/octet-stream')
            if description:
                data.add_field('description', description)
            
            # 添加API密钥头
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            
            async with self.session.post(url, data=data, headers=headers) as response:
                return await response.json()
                
        except Exception as e:
            self.logger.error(f"文件插入失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def documents_batch(self, files: List[Union[str, Path]]) -> Dict[str, Any]:
        """批量插入文件"""
        try:
            url = f"{self.base_url}/documents/batch"
            
            # 准备文件列表
            file_objects = []
            for file_path in files:
                file_path = Path(file_path)
                if file_path.exists():
                    file_objects.append(('files', (file_path.name, open(file_path, 'rb'), 'application/octet-stream')))
            
            if not file_objects:
                return {"status": "error", "message": "没有有效的文件"}
            
            # 发送请求
            if self.api_key:
                headers = {"X-API-Key": self.api_key}
            else:
                headers = {}
            
            response = requests.post(url, files=file_objects, headers=headers)
            
            # 关闭文件
            for _, file_tuple in file_objects:
                file_tuple[1].close()
            
            return response.json()
                
        except Exception as e:
            self.logger.error(f"批量插入失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def documents_scan(self) -> Dict[str, Any]:
        """扫描并索引新文档"""
        try:
            url = f"{self.base_url}/documents/scan"
            async with self.session.post(url, headers=self._get_headers()) as response:
                return await response.json()
        except Exception as e:
            self.logger.error(f"扫描文档失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def documents_scan_progress(self) -> Dict[str, Any]:
        """获取扫描进度"""
        try:
            url = f"{self.base_url}/documents/scan-progress"
            async with self.session.get(url, headers=self._get_headers()) as response:
                return await response.json()
        except Exception as e:
            self.logger.error(f"获取扫描进度失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def documents_upload(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """上传文件到input目录并索引"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return {"status": "error", "message": f"文件不存在: {file_path}"}
            
            url = f"{self.base_url}/documents/upload"
            
            with open(file_path, 'rb') as f:
                files = {'file': (file_path.name, f, 'application/octet-stream')}
                
                if self.api_key:
                    headers = {"X-API-Key": self.api_key}
                else:
                    headers = {}
                
                response = requests.post(url, files=files, headers=headers)
                return response.json()
                
        except Exception as e:
            self.logger.error(f"上传到input目录失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def documents(self) -> Dict[str, Any]:
        """获取已索引的文档列表"""
        try:
            url = f"{self.base_url}/documents"
            async with self.session.get(url, headers=self._get_headers()) as response:
                return await response.json()
        except Exception as e:
            self.logger.error(f"获取文档列表失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def documents_delete(self) -> Dict[str, Any]:
        """清空所有文档"""
        try:
            url = f"{self.base_url}/documents"
            async with self.session.delete(url, headers=self._get_headers()) as response:
                return await response.json()
        except Exception as e:
            self.logger.error(f"清空文档失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def upload_file(self, file_path: Union[str, Path], description: str = "") -> Dict[str, Any]:
        """上传文件到uploads目录（后台处理）"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return {"status": "error", "message": f"文件不存在: {file_path}"}
            
            url = f"{self.base_url}/upload/file"
            
            with open(file_path, 'rb') as f:
                files = {'file': (file_path.name, f, 'application/octet-stream')}
                data = {'description': description}
                
                if self.api_key:
                    headers = {"X-API-Key": self.api_key}
                else:
                    headers = {}
                
                response = requests.post(url, files=files, data=data, headers=headers)
                return response.json()
                
        except Exception as e:
            self.logger.error(f"文件上传失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def upload_image(self, image_path: Union[str, Path], description: str = "") -> Dict[str, Any]:
        """上传并解析图片（多模态理解）"""
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                return {"status": "error", "message": f"图片不存在: {image_path}"}
            
            url = f"{self.base_url}/upload/image"
            
            with open(image_path, 'rb') as f:
                files = {'image': (image_path.name, f, 'image/jpeg')}
                data = {'description': description}
                
                if self.api_key:
                    headers = {"X-API-Key": self.api_key}
                else:
                    headers = {}
                
                response = requests.post(url, files=files, data=data, headers=headers)
                return response.json()
                
        except Exception as e:
            self.logger.error(f"图片上传失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def query_image(self, image_path: Union[str, Path], question: str, mode: str = "mini") -> Dict[str, Any]:
        """基于图片的问答"""
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                return {"status": "error", "message": f"图片不存在: {image_path}"}
            
            url = f"{self.base_url}/query/image"
            
            with open(image_path, 'rb') as f:
                files = {'image': (image_path.name, f, 'image/jpeg')}
                data = {'question': question, 'mode': mode}
                
                if self.api_key:
                    headers = {"X-API-Key": self.api_key}
                else:
                    headers = {}
                
                response = requests.post(url, files=files, data=data, headers=headers)
                return response.json()
                
        except Exception as e:
            self.logger.error(f"图片问答失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def files_status(self) -> Dict[str, Any]:
        """获取文件处理状态"""
        try:
            url = f"{self.base_url}/files/status"
            async with self.session.get(url, headers=self._get_headers()) as response:
                return await response.json()
        except Exception as e:
            self.logger.error(f"获取文件状态失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def supported_formats(self) -> Dict[str, Any]:
        """获取支持的文件格式"""
        try:
            url = f"{self.base_url}/supported-formats"
            async with self.session.get(url, headers=self._get_headers()) as response:
                return await response.json()
        except Exception as e:
            self.logger.error(f"获取支持格式失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def files_delete(self, file_name: str) -> Dict[str, Any]:
        """删除指定文件"""
        try:
            url = f"{self.base_url}/files/{file_name}"
            async with self.session.delete(url, headers=self._get_headers()) as response:
                return await response.json()
        except Exception as e:
            self.logger.error(f"删除文件失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def system_info(self) -> Dict[str, Any]:
        """获取系统详细信息"""
        try:
            url = f"{self.base_url}/system/info"
            async with self.session.get(url, headers=self._get_headers()) as response:
                return await response.json()
        except Exception as e:
            self.logger.error(f"获取系统信息失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def files_batch_process(self, directory: str, file_types: str = "all") -> Dict[str, Any]:
        """批量处理指定目录中的文件"""
        try:
            url = f"{self.base_url}/files/batch-process"
            data = {'directory': directory, 'file_types': file_types}
            
            if self.api_key:
                headers = {"X-API-Key": self.api_key}
            else:
                headers = {}
            
            response = requests.post(url, data=data, headers=headers)
            return response.json()
                
        except Exception as e:
            self.logger.error(f"批量处理文件失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def files_search(self, query: str, file_type: str = None, size_min: int = None, size_max: int = None) -> Dict[str, Any]:
        """搜索文件"""
        try:
            url = f"{self.base_url}/files/search"
            params = {'query': query}
            if file_type:
                params['file_type'] = file_type
            if size_min:
                params['size_min'] = size_min
            if size_max:
                params['size_max'] = size_max
            
            async with self.session.get(url, params=params, headers=self._get_headers()) as response:
                return await response.json()
        except Exception as e:
            self.logger.error(f"搜索文件失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def api_endpoints(self) -> Dict[str, Any]:
        """获取所有可用的API端点信息"""
        try:
            url = f"{self.base_url}/api/endpoints"
            async with self.session.get(url, headers=self._get_headers()) as response:
                return await response.json()
        except Exception as e:
            self.logger.error(f"获取API端点失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def api_version(self) -> Dict[str, Any]:
        """获取Ollama版本信息"""
        try:
            url = f"{self.base_url}/api/version"
            async with self.session.get(url, headers=self._get_headers()) as response:
                return await response.json()
        except Exception as e:
            self.logger.error(f"获取版本信息失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def api_tags(self) -> Dict[str, Any]:
        """获取可用模型"""
        try:
            url = f"{self.base_url}/api/tags"
            async with self.session.get(url, headers=self._get_headers()) as response:
                return await response.json()
        except Exception as e:
            self.logger.error(f"获取模型标签失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def api_generate(self, prompt: str, system: str = None, stream: bool = False) -> Dict[str, Any]:
        """生成文本（Ollama兼容）"""
        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": "minirag:latest",
                "prompt": prompt,
                "stream": stream
            }
            if system:
                payload["system"] = system
            
            async with self.session.post(url, json=payload, headers=self._get_headers()) as response:
                return await response.json()
        except Exception as e:
            self.logger.error(f"生成文本失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def api_chat(self, messages: List[Dict[str, str]], stream: bool = False) -> Dict[str, Any]:
        """聊天对话（Ollama兼容）"""
        try:
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": "minirag:latest",
                "messages": messages,
                "stream": stream
            }
            
            async with self.session.post(url, json=payload, headers=self._get_headers()) as response:
                return await response.json()
        except Exception as e:
            self.logger.error(f"聊天对话失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def graph_label_list(self) -> Dict[str, Any]:
        """获取图形标签列表"""
        try:
            url = f"{self.base_url}/graph/label/list"
            async with self.session.get(url, headers=self._get_headers()) as response:
                return await response.json()
        except Exception as e:
            self.logger.error(f"获取图形标签失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def graphs(self, label: str, max_depth: int = 100) -> Dict[str, Any]:
        """获取图形"""
        try:
            url = f"{self.base_url}/graphs"
            params = {'label': label, 'max_depth': max_depth}
            async with self.session.get(url, params=params, headers=self._get_headers()) as response:
                return await response.json()
        except Exception as e:
            self.logger.error(f"获取图形失败: {e}")
            return {"status": "error", "message": str(e)}
    
    # 为了向后兼容，保留一些别名方法
    async def health_check(self) -> Dict[str, Any]:
        """健康检查（别名：health）"""
        return await self.health()
    
    async def insert_text(self, text: str, description: str = "") -> Dict[str, Any]:
        """插入文本到RAG系统（别名：documents_text）"""
        return await self.documents_text(text, description)
    
    async def upload_document(self, file_path: Union[str, Path], description: str = "") -> Dict[str, Any]:
        """上传并索引文档（别名：documents_file）"""
        return await self.documents_file(file_path, description)
    
    async def insert_file(self, file_path: Union[str, Path], description: str = "") -> Dict[str, Any]:
        """插入文件到RAG系统（别名：documents_file）"""
        return await self.documents_file(file_path, description)
    
    async def get_documents(self) -> Dict[str, Any]:
        """获取已索引的文档列表（别名：documents）"""
        return await self.documents()
    
    async def scan_documents(self) -> Dict[str, Any]:
        """扫描并索引新文档（别名：documents_scan）"""
        return await self.documents_scan()
    
    async def clear_documents(self) -> Dict[str, Any]:
        """清空所有文档（别名：documents_delete）"""
        return await self.documents_delete()
