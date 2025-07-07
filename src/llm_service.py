"""
LLM 服务模块

封装所有与 LLM API 交互的功能，包括文本生成、函数调用等。
"""
import logging
from typing import Dict, Any, Optional, List, Union
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logger = logging.getLogger("llm_service")

class LLMService:
    """LLM 服务类，封装所有与 LLM 交互的功能"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-ai/DeepSeek-R1"):
        """
        初始化 LLM 服务
        
        Args:
            api_key: OpenAI API 密钥，如果为 None 则从环境变量中读取
            model: 默认使用的模型
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY in .env file.")
            
        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key, 
                base_url="https://api.siliconflow.cn/v1")
    
    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成长度
            **kwargs: 其他参数
            
        Returns:
            生成的文本
        """
        messages: List[ChatCompletionMessageParam] = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("No content in response")
                
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise
    
    async def function_call(
        self,
        function_name: str,
        function_description: str,
        function_parameters: Dict[str, Any],
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行函数调用
        
        Args:
            function_name: 函数名
            function_description: 函数描述
            function_parameters: 函数参数定义
            prompt: 用户提示
            model: 使用的模型名称，如果为 None 则使用默认模型
            system_prompt: 系统提示
            **kwargs: 其他参数
            
        Returns:
            包含函数调用结果的字典
        """
        messages: List[ChatCompletionMessageParam] = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": prompt})
        
        functions = [{
            "type": "function",
            "function": {
                "name": function_name,
                "description": function_description,
                "parameters": function_parameters
            }
        }]
        
        # 使用传入的模型或默认模型
        model_to_use = model or self.model
        
        try:
            response = await self.client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                tools=functions,
                tool_choice={"type": "function", "function": {"name": function_name}},
                **kwargs
            )
            
            message = response.choices[0].message
            tool_calls = message.tool_calls if hasattr(message, 'tool_calls') else []
            
            if tool_calls:
                function_call = tool_calls[0].function
                return {
                    "function_name": function_call.name,
                    "arguments": function_call.arguments,
                    "message": message.content,
                    "model": self.model,
                    "usage": response.usage.model_dump() if hasattr(response, 'usage') else None
                }
            else:
                return {
                    "message": message.content,
                    "model": self.model,
                    "usage": response.usage.model_dump() if hasattr(response, 'usage') else None
                }
                
        except Exception as e:
            logger.error(f"Error in function call: {str(e)}")
            raise
    
    async def get_embeddings(
        self,
        texts: Union[str, List[str]],
        model: str = "text-embedding-3-small"
    ) -> List[List[float]]:
        """
        获取文本嵌入
        
        Args:
            texts: 单个文本或文本列表
            model: 嵌入模型
            
        Returns:
            嵌入向量列表
        """
        if isinstance(texts, str):
            texts = [texts]
            
        try:
            response = await self.client.embeddings.create(
                input=texts,
                model=model
            )
            
            return [item.embedding for item in response.data]
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            raise

# 全局 LLM 服务实例
llm_service = LLMService()
