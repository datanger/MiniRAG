"""
文件监控和自动解析服务模块
支持定时监控文件夹，自动解析新文件
"""

import os
import time
import asyncio
import logging
import threading
from pathlib import Path
from typing import Dict, List, Set, Optional, Callable, Any
from datetime import datetime, timedelta
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import json
import configparser

from .file_parser import FileParserManager, parse_single_file
from .minirag import MiniRAG

logger = logging.getLogger(__name__)


class FileChangeHandler(FileSystemEventHandler):
    """文件变化事件处理器"""
    
    def __init__(self, callback: Callable[[str], None]):
        """初始化处理器"""
        self.callback = callback
        self.processed_files: Set[str] = set()
    
    def on_created(self, event):
        """文件创建事件"""
        if not event.is_directory:
            file_path = event.src_path
            if file_path not in self.processed_files:
                self.processed_files.add(file_path)
                self.callback(file_path)
    
    def on_modified(self, event):
        """文件修改事件"""
        if not event.is_directory:
            file_path = event.src_path
            if file_path not in self.processed_files:
                self.processed_files.add(file_path)
                self.callback(file_path)


class FileMonitor:
    """文件监控器"""
    
    def __init__(self, watch_paths: List[str], callback: Callable[[str], None]):
        """初始化文件监控器"""
        self.watch_paths = watch_paths
        self.callback = callback
        self.observer = Observer()
        self.handler = FileChangeHandler(callback)
        self.is_running = False
    
    def start(self):
        """启动文件监控"""
        try:
            for path in self.watch_paths:
                if os.path.exists(path):
                    self.observer.schedule(self.handler, path, recursive=True)
                    logger.info(f"开始监控路径: {path}")
                else:
                    logger.warning(f"监控路径不存在: {path}")
            
            self.observer.start()
            self.is_running = True
            logger.info("文件监控器已启动")
        except Exception as e:
            logger.error(f"启动文件监控器失败: {e}")
    
    def stop(self):
        """停止文件监控"""
        if self.is_running:
            self.observer.stop()
            self.observer.join()
            self.is_running = False
            logger.info("文件监控器已停止")
    
    def is_active(self) -> bool:
        """检查监控器是否活跃"""
        return self.is_running


class AutoFileProcessor:
    """自动文件处理器"""
    
    def __init__(self, minirag_instance: MiniRAG, config: Dict[str, Any]):
        """初始化自动文件处理器"""
        self.minirag = minirag_instance
        self.config = config
        self.file_parser = FileParserManager()
        self.processed_files: Set[str] = set()
        self.failed_files: Dict[str, str] = {}
        
        # 从配置获取设置
        self.watch_paths = config.get('watch_paths', [])
        self.scan_interval = config.get('scan_interval', 300)  # 默认5分钟
        self.file_extensions = config.get('file_extensions', [])
        self.max_file_size = config.get('max_file_size', 100 * 1024 * 1024)  # 默认100MB
        self.enable_realtime_monitor = config.get('enable_realtime_monitor', True)
        
        # 文件监控器
        self.file_monitor: Optional[FileMonitor] = None
        self.monitor_thread: Optional[threading.Thread] = None
        
        # 定时扫描任务
        self.scan_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动自动文件处理服务"""
        logger.info("启动自动文件处理服务")
        
        # 启动实时文件监控
        if self.enable_realtime_monitor:
            await self._start_file_monitor()
        
        # 启动定时扫描
        await self._start_periodic_scan()
        
        logger.info("自动文件处理服务已启动")
    
    async def stop(self):
        """停止自动文件处理服务"""
        logger.info("停止自动文件处理服务")
        
        # 停止文件监控
        if self.file_monitor:
            self.file_monitor.stop()
        
        # 停止定时扫描
        if self.scan_task and not self.scan_task.done():
            self.scan_task.cancel()
        
        logger.info("自动文件处理服务已停止")
    
    async def _start_file_monitor(self):
        """启动文件监控"""
        if not self.watch_paths:
            return
        
        def file_change_callback(file_path: str):
            """文件变化回调函数"""
            asyncio.create_task(self._process_single_file(file_path))
        
        self.file_monitor = FileMonitor(self.watch_paths, file_change_callback)
        
        # 在单独的线程中启动监控器
        def run_monitor():
            self.file_monitor.start()
            try:
                while self.file_monitor.is_active():
                    time.sleep(1)
            except KeyboardInterrupt:
                self.file_monitor.stop()
        
        self.monitor_thread = threading.Thread(target=run_monitor, daemon=True)
        self.monitor_thread.start()
    
    async def _start_periodic_scan(self):
        """启动定时扫描"""
        async def periodic_scan():
            while True:
                try:
                    await self.scan_and_process_files()
                    await asyncio.sleep(self.scan_interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"定时扫描出错: {e}")
                    await asyncio.sleep(60)  # 出错后等待1分钟再试
        
        self.scan_task = asyncio.create_task(periodic_scan())
    
    async def scan_and_process_files(self):
        """扫描并处理文件"""
        logger.info("开始扫描文件...")
        
        all_files = []
        for watch_path in self.watch_paths:
            if os.path.exists(watch_path):
                files = self._scan_directory(watch_path)
                all_files.extend(files)
        
        # 过滤已处理的文件
        new_files = [f for f in all_files if f not in self.processed_files]
        
        if new_files:
            logger.info(f"发现 {len(new_files)} 个新文件")
            await self._process_files(new_files)
        else:
            logger.info("没有发现新文件")
    
    def _scan_directory(self, directory: str) -> List[str]:
        """扫描目录中的文件"""
        files = []
        try:
            for root, dirs, filenames in os.walk(directory):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    
                    # 检查文件扩展名
                    if self.file_extensions:
                        if not any(filename.lower().endswith(ext.lower()) for ext in self.file_extensions):
                            continue
                    
                    # 检查文件大小
                    try:
                        file_size = os.path.getsize(file_path)
                        if file_size > self.max_file_size:
                            logger.warning(f"文件过大，跳过: {file_path} ({file_size} bytes)")
                            continue
                    except OSError:
                        continue
                    
                    files.append(file_path)
        except Exception as e:
            logger.error(f"扫描目录失败 {directory}: {e}")
        
        return files
    
    async def _process_files(self, file_paths: List[str]):
        """批量处理文件"""
        logger.info(f"开始处理 {len(file_paths)} 个文件")
        
        for file_path in file_paths:
            try:
                await self._process_single_file(file_path)
            except Exception as e:
                logger.error(f"处理文件失败 {file_path}: {e}")
                self.failed_files[file_path] = str(e)
    
    async def _process_single_file(self, file_path: str):
        """处理单个文件"""
        if file_path in self.processed_files:
            return
        
        logger.info(f"处理文件: {file_path}")
        
        try:
            # 解析文件
            parsed_result = await parse_single_file(file_path)
            
            if 'error' in parsed_result:
                logger.error(f"解析文件失败 {file_path}: {parsed_result['error']}")
                self.failed_files[file_path] = parsed_result['error']
                return
            
            # 检查是否有内容
            if not parsed_result.get('content'):
                logger.warning(f"文件内容为空: {file_path}")
                return
            
            # 插入到MiniRAG
            await self.minirag.ainsert(parsed_result['content'])
            
            # 标记为已处理
            self.processed_files.add(file_path)
            
            # 从失败列表中移除
            if file_path in self.failed_files:
                del self.failed_files[file_path]
            
            logger.info(f"文件处理成功: {file_path}")
            
        except Exception as e:
            logger.error(f"处理文件失败 {file_path}: {e}")
            self.failed_files[file_path] = str(e)
    
    def get_status(self) -> Dict[str, Any]:
        """获取处理状态"""
        return {
            'is_running': self.file_monitor.is_active() if self.file_monitor else False,
            'processed_files_count': len(self.processed_files),
            'failed_files_count': len(self.failed_files),
            'watch_paths': self.watch_paths,
            'scan_interval': self.scan_interval,
            'file_extensions': self.file_extensions,
            'max_file_size': self.max_file_size,
            'enable_realtime_monitor': self.enable_realtime_monitor,
            'processed_files': list(self.processed_files),
            'failed_files': self.failed_files.copy()
        }
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """获取支持的文件格式"""
        return self.file_parser.get_supported_formats()


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: str = "file_processor_config.ini"):
        """初始化配置管理器"""
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_file):
            self.config.read(self.config_file, encoding='utf-8')
        else:
            self._create_default_config()
    
    def _create_default_config(self):
        """创建默认配置"""
        self.config['DEFAULT'] = {
            'scan_interval': '300',
            'max_file_size': '104857600',
            'enable_realtime_monitor': 'true'
        }
        
        self.config['WATCH_PATHS'] = {
            'path1': './documents',
            'path2': './uploads'
        }
        
        self.config['FILE_EXTENSIONS'] = {
            'extensions': '.pdf,.docx,.txt,.md,.pptx,.png,.jpg,.jpeg'
        }
        
        self.config['MINIRAG'] = {
            'working_dir': './minirag_cache',
            'chunk_token_size': '1200',
            'chunk_overlap_token_size': '100'
        }
        
        self.config['LLM'] = {
            'model_name': 'microsoft/Phi-3.5-mini-instruct',
            'max_token_size': '2000'
        }
        
        self.config['EMBEDDING'] = {
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'embedding_dim': '384'
        }
        
        self._save_config()
    
    def _save_config(self):
        """保存配置文件"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            self.config.write(f)
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置字典"""
        config_dict = {}
        
        # 获取监控路径
        watch_paths = []
        for key, value in self.config['WATCH_PATHS'].items():
            if key.startswith('path'):
                watch_paths.append(value)
        config_dict['watch_paths'] = watch_paths
        
        # 获取文件扩展名
        extensions_str = self.config['FILE_EXTENSIONS'].get('extensions', '')
        config_dict['file_extensions'] = [ext.strip() for ext in extensions_str.split(',') if ext.strip()]
        
        # 获取其他配置
        config_dict['scan_interval'] = int(self.config['DEFAULT'].get('scan_interval', '300'))
        config_dict['max_file_size'] = int(self.config['DEFAULT'].get('max_file_size', '104857600'))
        config_dict['enable_realtime_monitor'] = self.config['DEFAULT'].getboolean('enable_realtime_monitor', True)
        
        # MiniRAG配置
        config_dict['minirag'] = {
            'working_dir': self.config['MINIRAG'].get('working_dir', './minirag_cache'),
            'chunk_token_size': int(self.config['MINIRAG'].get('chunk_token_size', '1200')),
            'chunk_overlap_token_size': int(self.config['MINIRAG'].get('chunk_overlap_token_size', '100'))
        }
        
        # LLM配置
        config_dict['llm'] = {
            'model_name': self.config['LLM'].get('model_name', 'microsoft/Phi-3.5-mini-instruct'),
            'max_token_size': int(self.config['LLM'].get('max_token_size', '2000'))
        }
        
        # 嵌入模型配置
        config_dict['embedding'] = {
            'model_name': self.config['EMBEDDING'].get('model_name', 'sentence-transformers/all-MiniLM-L6-v2'),
            'embedding_dim': int(self.config['EMBEDDING'].get('embedding_dim', '384'))
        }
        
        # 多模态模型配置
        config_dict['multimodal'] = {
            'model_type': self.config['MULTIMODAL'].get('model_type', 'none'),
            'openai_api_key': self.config['MULTIMODAL'].get('openai_api_key', ''),
            'openai_model': self.config['MULTIMODAL'].get('openai_model', 'gpt-4-vision-preview'),
            'llava_model_path': self.config['MULTIMODAL'].get('llava_model_path', 'liuhaotian/llava-v1.5-7b'),
            'qwen_vl_model_path': self.config['MULTIMODAL'].get('qwen_vl_model_path', 'Qwen/Qwen-VL-Chat'),
            'enable_multimodal': self.config['MULTIMODAL'].getboolean('enable_multimodal', False)
        }
        
        return config_dict
    
    def update_config(self, section: str, key: str, value: str):
        """更新配置"""
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
        self._save_config()
    
    def reload_config(self):
        """重新加载配置"""
        self._load_config()


# 便捷函数
async def create_auto_processor(config_file: str = "file_processor_config.ini") -> AutoFileProcessor:
    """创建自动文件处理器的便捷函数"""
    config_manager = ConfigManager(config_file)
    config = config_manager.get_config()
    
    # 这里需要根据配置创建MiniRAG实例
    # 由于依赖关系，这里只是示例
    from .minirag import MiniRAG
    minirag = MiniRAG(
        working_dir=config['minirag']['working_dir'],
        chunk_token_size=config['minirag']['chunk_token_size'],
        chunk_overlap_token_size=config['minirag']['chunk_overlap_token_size']
    )
    
    return AutoFileProcessor(minirag, config)
