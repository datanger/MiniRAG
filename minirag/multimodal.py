"""
多模态模型接口和实现
支持图片理解和多模态问答
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from pathlib import Path
import base64
import io

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class MultimodalModel(ABC):
    """多模态模型基类"""
    
    @abstractmethod
    async def understand_image(self, image: Union[str, Path, Image.Image], prompt: str) -> str:
        """理解图片内容"""
        pass
    
    @abstractmethod
    async def query_with_image(self, question: str, image: Union[str, Path, Image.Image]) -> str:
        """基于图片进行问答"""
        pass
    
    @abstractmethod
    def can_handle_image(self, image: Union[str, Path, Image.Image]) -> bool:
        """检查是否可以处理该图片"""
        pass


class OpenAIVisionModel(MultimodalModel):
    """OpenAI Vision模型实现"""
    
    def __init__(self, api_key: str, model: str = "gpt-4-vision-preview"):
        """初始化OpenAI Vision模型
        
        Args:
            api_key: OpenAI API密钥
            model: 使用的模型名称
        """
        self.api_key = api_key
        self.model = model
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """初始化OpenAI客户端"""
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            logger.error("OpenAI库未安装，请运行: pip install openai")
            raise ImportError("OpenAI库未安装")
    
    def can_handle_image(self, image: Union[str, Path, Image.Image]) -> bool:
        """检查是否可以处理该图片"""
        if not PIL_AVAILABLE:
            return False
        
        try:
            if isinstance(image, (str, Path)):
                img = Image.open(image)
            else:
                img = image
            
            # 检查图片格式和大小
            if img.format not in ['JPEG', 'PNG', 'GIF', 'WEBP']:
                return False
            
            # 检查图片尺寸（OpenAI Vision有尺寸限制）
            max_size = 2048
            if img.width > max_size or img.height > max_size:
                return False
            
            return True
        except Exception:
            return False
    
    async def understand_image(self, image: Union[str, Path, Image.Image], prompt: str) -> str:
        """理解图片内容"""
        try:
            # 准备图片数据
            image_data = await self._prepare_image(image)
            
            # 调用OpenAI Vision API
            response = await self._call_vision_api(image_data, prompt)
            return response
            
        except Exception as e:
            logger.error(f"OpenAI Vision理解图片失败: {e}")
            raise
    
    async def query_with_image(self, question: str, image: Union[str, Path, Image.Image]) -> str:
        """基于图片进行问答"""
        try:
            # 准备图片数据
            image_data = await self._prepare_image(image)
            
            # 构建完整的提示词
            full_prompt = f"用户问题：{question}\n\n请基于这张图片回答问题，如果图片中没有相关信息，请说明。"
            
            # 调用OpenAI Vision API
            response = await self._call_vision_api(image_data, full_prompt)
            return response
            
        except Exception as e:
            logger.error(f"OpenAI Vision图片问答失败: {e}")
            raise
    
    async def _prepare_image(self, image: Union[str, Path, Image.Image]) -> str:
        """准备图片数据为base64格式"""
        try:
            if isinstance(image, (str, Path)):
                img = Image.open(image)
            else:
                img = image
            
            # 转换为RGB模式（如果需要）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 调整图片大小（如果需要）
            max_size = 2048
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # 转换为base64
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=95)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return img_base64
            
        except Exception as e:
            logger.error(f"准备图片数据失败: {e}")
            raise
    
    async def _call_vision_api(self, image_data: str, prompt: str) -> str:
        """调用OpenAI Vision API"""
        try:
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ]
            
            # 调用API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"调用OpenAI Vision API失败: {e}")
            raise


class LLaVAModel(MultimodalModel):
    """LLaVA模型实现"""
    
    def __init__(self, model_path: str = "liuhaotian/llava-v1.5-7b"):
        """初始化LLaVA模型
        
        Args:
            model_path: 模型路径或名称
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._init_model()
    
    def _init_model(self):
        """初始化LLaVA模型"""
        try:
            import torch
            from transformers import LlavaForConditionalGeneration, LlavaProcessor
            from PIL import Image
            
            self.processor = LlavaProcessor.from_pretrained(self.model_path)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map="auto"
            )
            
        except ImportError:
            logger.error("Transformers或torch库未安装，请运行: pip install transformers torch")
            raise ImportError("Transformers或torch库未安装")
        except Exception as e:
            logger.error(f"初始化LLaVA模型失败: {e}")
            raise
    
    def can_handle_image(self, image: Union[str, Path, Image.Image]) -> bool:
        """检查是否可以处理该图片"""
        if not PIL_AVAILABLE or self.model is None:
            return False
        
        try:
            if isinstance(image, (str, Path)):
                img = Image.open(image)
            else:
                img = image
            
            # LLaVA支持常见的图片格式
            if img.format not in ['JPEG', 'PNG', 'GIF', 'WEBP']:
                return False
            
            return True
        except Exception:
            return False
    
    async def understand_image(self, image: Union[str, Path, Image.Image], prompt: str) -> str:
        """理解图片内容"""
        try:
            # 准备图片
            if isinstance(image, (str, Path)):
                img = Image.open(image)
            else:
                img = image
            
            # 构建提示词
            full_prompt = f"USER: {prompt}\nASSISTANT:"
            
            # 处理输入
            inputs = self.processor(
                text=full_prompt,
                images=img,
                return_tensors="pt"
            )
            
            # 生成回答
            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            # 解码输出
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # 提取助手回答部分
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"LLaVA理解图片失败: {e}")
            raise
    
    async def query_with_image(self, question: str, image: Union[str, Path, Image.Image]) -> str:
        """基于图片进行问答"""
        try:
            # 准备图片
            if isinstance(image, (str, Path)):
                img = Image.open(image)
            else:
                img = image
            
            # 构建提示词
            full_prompt = f"USER: 请基于这张图片回答问题：{question}\nASSISTANT:"
            
            # 处理输入
            inputs = self.processor(
                text=full_prompt,
                images=img,
                return_tensors="pt"
            )
            
            # 生成回答
            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            # 解码输出
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # 提取助手回答部分
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"LLaVA图片问答失败: {e}")
            raise


class QwenVLModel(MultimodalModel):
    """Qwen-VL模型实现"""
    
    def __init__(self, model_path: str = "Qwen/Qwen-VL-Chat"):
        """初始化Qwen-VL模型
        
        Args:
            model_path: 模型路径或名称
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._init_model()
    
    def _init_model(self):
        """初始化Qwen-VL模型"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from PIL import Image
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                trust_remote_code=True
            )
            
        except ImportError:
            logger.error("Transformers库未安装，请运行: pip install transformers")
            raise ImportError("Transformers库未安装")
        except Exception as e:
            logger.error(f"初始化Qwen-VL模型失败: {e}")
            raise
    
    def can_handle_image(self, image: Union[str, Path, Image.Image]) -> bool:
        """检查是否可以处理该图片"""
        if not PIL_AVAILABLE or self.model is None:
            return False
        
        try:
            if isinstance(image, (str, Path)):
                img = Image.open(image)
            else:
                img = image
            
            # Qwen-VL支持常见的图片格式
            if img.format not in ['JPEG', 'PNG', 'GIF', 'WEBP']:
                return False
            
            return True
        except Exception:
            return False
    
    async def understand_image(self, image: Union[str, Path, Image.Image], prompt: str) -> str:
        """理解图片内容"""
        try:
            # 准备图片
            if isinstance(image, (str, Path)):
                img = Image.open(image)
            else:
                img = image
            
            # 构建提示词
            full_prompt = f"<image>\n{prompt}"
            
            # 处理输入
            inputs = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": full_prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 编码输入
            input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids.to(self.model.device)
            
            # 生成回答
            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            # 解码输出
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取回答部分
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Qwen-VL理解图片失败: {e}")
            raise
    
    async def query_with_image(self, question: str, image: Union[str, Path, Image.Image]) -> str:
        """基于图片进行问答"""
        try:
            # 准备图片
            if isinstance(image, (str, Path)):
                img = Image.open(image)
            else:
                img = image
            
            # 构建提示词
            full_prompt = f"<image>\n请基于这张图片回答问题：{question}"
            
            # 处理输入
            inputs = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": full_prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 编码输入
            input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids.to(self.model.device)
            
            # 生成回答
            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            # 解码输出
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取回答部分
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Qwen-VL图片问答失败: {e}")
            raise


# 便捷函数
def create_multimodal_model(model_type: str, **kwargs) -> MultimodalModel:
    """创建多模态模型的便捷函数
    
    Args:
        model_type: 模型类型 ('openai', 'llava', 'qwen-vl')
        **kwargs: 模型特定参数
    
    Returns:
        多模态模型实例
    """
    if model_type.lower() == 'openai':
        if 'api_key' not in kwargs:
            raise ValueError("OpenAI模型需要提供api_key参数")
        return OpenAIVisionModel(**kwargs)
    
    elif model_type.lower() == 'llava':
        return LLaVAModel(**kwargs)
    
    elif model_type.lower() == 'qwen-vl':
        return QwenVLModel(**kwargs)
    
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


# 异步包装器
async def async_multimodal_wrapper(model: MultimodalModel, method: str, *args, **kwargs):
    """异步包装器，用于同步模型方法"""
    if asyncio.iscoroutinefunction(getattr(model, method)):
        return await getattr(model, method)(*args, **kwargs)
    else:
        # 在线程池中运行同步方法
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: getattr(model, method)(*args, **kwargs)
        )
