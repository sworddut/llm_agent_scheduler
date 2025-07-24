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
import json
from fastmcp import Client

# read JSON config
with open("src/mcp/mcp_config.json", "r") as f:
    config = json.load(f)

# 加载环境变量
load_dotenv()

# 配置日志
logger = logging.getLogger("llm_service")

class LLMService:
    """LLM 服务类，封装所有与 LLM 交互的功能"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
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
        self.mcp_client = Client(config)
        self.client = AsyncOpenAI(api_key=self.api_key, 
                base_url=os.getenv("OPENAI_BASE_URL"))
    
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str],
        **kwargs
    ):
        """
        Creates a model response for the given chat conversation.
        This is a wrapper around openai.chat.completions.create.
        """
        if model is None:
            model = self.model
        
        # 获取可用工具列表并格式化为OpenAI API兼容的格式
        async with self.mcp_client:
            mcp_tools = await self.mcp_client.list_tools()
        
        # The 'tools' parameter expects a list of dicts, not a JSON string.
        # We also need to rename 'inputSchema' to 'parameters' for OpenAI compatibility.
        tools_for_openai = []
        for tool in mcp_tools:
            tool_as_dict = tool.__dict__
            tool_as_dict['parameters'] = tool_as_dict.pop('inputSchema', {})
            tools_for_openai.append({
                "type": "function",
                "function": tool_as_dict
            })

        logger.debug(f"Sending chat completion request to model {model} with {len(messages)} messages.")
        try:
            api_kwargs = kwargs.copy()
            if tools_for_openai and "response_format" not in api_kwargs:
                api_kwargs["tools"] = tools_for_openai
                api_kwargs["tool_choice"] = "auto"
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                **api_kwargs
            )
            logger.debug("Received chat completion response.")
            return response
        except Exception as e:
            logger.error(f"An error occurred during chat completion: {e}")
            raise

    async def process_and_call_tool(self, tool_call) -> str:
        """
        Processes a single tool call, executes it, and returns the result as a JSON string.
        
        :param tool_call: A tool call object from the OpenAI response.
        :return: A JSON string representing the result of the tool execution.
        """
        try:
            tool_name = tool_call.function.name
            tool_args_str = tool_call.function.arguments
            logger.info(f"Processing tool call: {tool_name} with args: {tool_args_str}")

            # Parse the arguments string into a dictionary
            tool_args = json.loads(tool_args_str)

            # Call the tool using the MCP client
            async with self.mcp_client as client:
                logger.info(f"Executing tool '{tool_name}' via MCP client.")
                result = await client.call_tool(tool_name, tool_args)
                logger.info(f"Tool '{tool_name}' executed. Result: {result}")
                
                # Ensure the result is a JSON string for the next LLM call
                if isinstance(result, (dict, list)):
                    return json.dumps(result)
                return str(result)

        except (AttributeError, json.JSONDecodeError) as e:
            error_message = f"Error processing tool call: {e}"
            logger.error(error_message, exc_info=True)
            return json.dumps({"error": error_message})
    
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
        print('api_key:',self.api_key)
        print('base_url:',os.getenv("OPENAI_BASE_URL"))
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=60.0,  # Add a 60-second timeout
                **kwargs
            )
            
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("No content in response")
            print('llm output:',response.choices[0].message.content)
            return response.choices[0].message.content
            
        except asyncio.TimeoutError:
            logger.error("LLM API call timed out after 60 seconds.")
            raise
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
            print('function_call message:',message)
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
