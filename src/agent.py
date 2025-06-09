# src/agent.py
import asyncio
import logging
import os
import time
import random
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from .scheduler import Scheduler, SchedulingStrategy
from .task import Task, TaskStatus, TaskType
from .llm_service import LLMService

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Agent")

class Agent:
    def __init__(self, scheduler: Scheduler, max_concurrent_tasks: int = 3):
        self.scheduler = scheduler
        self.running = False
        self.max_concurrent_tasks = max_concurrent_tasks
        self.active_tasks = 0
        self.llm_service = LLMService()
        self.task_handlers = {}
        self.register_default_handlers()
    
    def register_default_handlers(self):
        """注册默认任务处理器"""
        self.task_handlers[TaskType.FUNCTION_CALL] = self.handle_function_call
        self.task_handlers[TaskType.API_REQUEST] = self.handle_api_request
        self.task_handlers[TaskType.FILE_OPERATION] = self.handle_file_operation
        self.task_handlers["generate_text"] = self.handle_generate_text  # 添加对 generate_text 任务类型的支持
    
    def register_handler(self, task_type: TaskType, handler: Callable):
        """注册自定义任务处理器"""
        self.task_handlers[task_type] = handler
    
    async def start(self):
        """启动代理"""
        self.running = True
        logger.info("Agent started with max concurrent tasks: %d", self.max_concurrent_tasks)
        
        # 创建任务处理协程
        worker_tasks = [self.worker() for _ in range(self.max_concurrent_tasks)]
        await asyncio.gather(*worker_tasks)
    
    async def worker(self):
        """工作协程，负责处理任务"""
        while self.running:
            if self.active_tasks < self.max_concurrent_tasks:
                task = await self.scheduler.get_next_task()
                if task:
                    self.active_tasks += 1
                    # 创建新的协程来处理任务，但不等待它完成
                    asyncio.create_task(self.process_task(task))
                else:
                    # 没有任务，等待一段时间
                    await asyncio.sleep(0.5)
            else:
                # 已达到最大并发数，等待
                await asyncio.sleep(0.1)
    
    async def process_task(self, task: Task):
        """处理单个任务的协程"""
        try:
            # 标记任务开始
            task.start()
            logger.info(f"⚙️ 执行任务: {task}")
            
            # 根据任务类型调用相应的处理器
            if task.task_type in self.task_handlers:
                handler = self.task_handlers[task.task_type]
                result = await handler(task)
                task.complete(result)
            else:
                # 未知任务类型，使用默认处理
                logger.warning(f"未知任务类型: {task.task_type}，使用默认处理")
                await asyncio.sleep(random.uniform(0.5, 1.5))  # 模拟处理时间
                task.complete(f"执行了未知类型任务 {task.name}")
            
            logger.info(f"✅ 完成任务: {task}")
        except Exception as e:
            logger.error(f"❌ 任务执行失败: {task}", exc_info=True)
            task.fail(str(e))
        finally:
            # 通知调度器任务已完成
            await self.scheduler.task_completed(task)
            self.active_tasks -= 1
    
    async def handle_function_call(self, task: Task) -> Any:
        """处理函数调用类型的任务"""
        try:
            # 从任务负载中提取必要信息
            function_name = task.payload.get("function_name", "")
            function_description = task.payload.get("description", f"执行{function_name}函数")
            function_parameters = task.payload.get("parameters", {"type": "object", "properties": {}})
            prompt = task.payload.get("content", "请执行指定的函数")
            system_prompt = task.payload.get("system_prompt", "你是一个AI助手，请根据用户的请求执行相应的功能。")
            model = task.payload.get("model")  # 不设置默认值，让 LLMService 使用其默认值
            
            # 记录调试信息
            logger.info(f"处理函数调用: {function_name}")
            logger.debug(f"函数描述: {function_description}")
            logger.debug(f"函数参数: {function_parameters}")
            logger.debug(f"提示: {prompt}")
            logger.debug(f"系统提示: {system_prompt}")
            logger.debug(f"模型: {model}")
            
            # 使用 LLMService 执行函数调用
            try:
                result = await self.llm_service.function_call(
                    function_name=function_name,
                    function_description=function_description,
                    function_parameters=function_parameters,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model
                )
                logger.info(f"函数调用成功: {function_name}")
                logger.debug(f"函数调用结果: {result}")
                return result
            except Exception as e:
                logger.error(f"函数调用执行失败: {str(e)}", exc_info=True)
                raise
            
        except Exception as e:
            logger.error(f"函数调用失败: {str(e)}", exc_info=True)
            raise
    
    async def handle_api_request(self, task: Task) -> Any:
        """处理API请求类型的任务"""
        # 这里可以实现实际的API请求逻辑
        # 目前只是模拟
        await asyncio.sleep(random.uniform(0.3, 1.0))
        return {
            "status": "success",
            "data": f"API请求结果: {task.name}",
            "timestamp": datetime.now().isoformat()
        }
    
    async def handle_file_operation(self, task: Task) -> Any:
        """处理文件操作类型的任务"""
        # 这里可以实现实际的文件操作逻辑
        # 目前只是模拟
        await asyncio.sleep(random.uniform(0.2, 0.8))
        return {
            "status": "success",
            "filename": task.payload.get("filename", "unknown.txt"),
            "operation": task.payload.get("operation", "read"),
            "size": random.randint(100, 10000),
            "timestamp": datetime.now().isoformat()
        }
        
    async def handle_generate_text(self, task: Task) -> Any:
        """处理文本生成任务"""
        try:
            # 从任务负载中提取必要信息
            prompt = task.payload.get("content", "")
            system_prompt = task.payload.get("system_prompt", "你是一个AI助手，请根据用户的请求执行相应的功能。")
            temperature = task.payload.get("temperature", 0.7)
            max_tokens = task.payload.get("max_tokens", 1000)
            model = task.payload.get("model")  # 不设置默认值，让 LLMService 使用其默认值
            
            # 记录调试信息
            logger.info(f"处理文本生成任务: {task.name}")
            logger.debug(f"提示: {prompt}")
            logger.debug(f"系统提示: {system_prompt}")
            logger.debug(f"温度: {temperature}, 最大token数: {max_tokens}")
            logger.debug(f"模型: {model}")
            
            # 使用 LLMService 生成文本
            try:
                result = await self.llm_service.generate_text(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=model
                )
                logger.info(f"文本生成成功: {task.name}")
                logger.debug(f"生成结果: {result}")
                return {
                    "status": "success",
                    "generated_text": result,
                    "model": model or self.llm_service.model,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"文本生成失败: {str(e)}", exc_info=True)
                raise
                
        except Exception as e:
            logger.error(f"处理文本生成任务时出错: {str(e)}", exc_info=True)
            raise
    
    async def stop(self):
        """停止代理"""
        logger.info("Agent stopping...")
        self.running = False
