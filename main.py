# main.py
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
import asyncio
import logging
import os
import time
from datetime import datetime

from src.task import Task, TaskStatus, TaskType
from src.scheduler import Scheduler, SchedulingStrategy
from src.agent import Agent

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main")

# 创建应用
app = FastAPI(
    title="LLM Agent Scheduler",
    description="一个受操作系统调度启发的 LLM Agent 异步任务调度系统",
    version="0.2.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境应该限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建调度器和代理
scheduler = Scheduler(levels=3, strategy=SchedulingStrategy.PRIORITY_BASED)
agent = Agent(scheduler, max_concurrent_tasks=3)

# 模型定义
from typing import Union, Optional, List, Dict, Any

class TaskInput(BaseModel):
    name: str = Field(..., description="任务名称")
    payload: Dict[str, Any] = Field(..., description="任务负载数据")
    priority: int = Field(1, description="任务优先级 (0: 高, 1: 中, 2: 低)")
    task_type: Union[TaskType, str] = Field(TaskType.FUNCTION_CALL, description="任务类型")
    estimated_time: float = Field(1.0, description="预估执行时间（秒）")
    tags: Optional[List[str]] = Field(None, description="任务标签")

class TaskResponse(BaseModel):
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    message: str = Field(..., description="状态消息")

class StrategyUpdate(BaseModel):
    strategy: SchedulingStrategy = Field(..., description="调度策略")
    time_slice: Optional[float] = Field(None, description="时间片大小（秒）")

# 启动事件
@app.on_event("startup")
async def startup_event():
    logger.info("Starting LLM Agent Scheduler...")
    asyncio.create_task(agent.start())

# 关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down LLM Agent Scheduler...")
    await agent.stop()

# API路由
@app.get("/")
async def root():
    return {
        "message": "LLM Agent Scheduler 正在运行 ",
        "version": "0.2.0",
        "docs_url": "/docs",
        "stats": await scheduler.get_stats()
    }

@app.post("/tasks", response_model=TaskResponse, tags=["Tasks"])
async def submit_task(input: TaskInput):
    """
    提交新任务到调度队列
    
    - **name**: 任务名称
    - **payload**: 任务负载数据
    - **priority**: 任务优先级 (0: 高, 1: 中, 2: 低)
    - **task_type**: 任务类型 (function_call, api_request, file_operation 或自定义类型)
    - **estimated_time**: 预估执行时间（秒）
    - **tags**: 任务标签
    """
    try:
        # 处理 task_type，确保它是字符串
        task_type = input.task_type.value if isinstance(input.task_type, TaskType) else input.task_type
        
        # 创建新任务
        task = Task(
            name=input.name,
            payload=input.payload,
            priority=input.priority,
            task_type=task_type,  # 使用处理后的 task_type
            estimated_time=input.estimated_time,
            tags=input.tags
        )
        
        # 将任务添加到调度器
        await scheduler.add_task(task)
        
        logger.info(f"任务已提交: {task.id} - {task.name} (类型: {task_type}, 优先级: {task.priority})")
        
        return {"task_id": task.id, "status": "queued", "message": "任务已加入队列"}
        
    except Exception as e:
        print(e)
        logger.error(f"提交任务失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/tasks", response_model=List[Dict[str, Any]])
async def list_tasks(
    status: Optional[TaskStatus] = Query(None, description="按状态筛选"),
    priority: Optional[int] = Query(None, description="按优先级筛选"),
    task_type: Optional[TaskType] = Query(None, description="按任务类型筛选"),
    limit: int = Query(10, description="返回结果数量限制")
):
    """获取任务列表"""
    # 获取所有任务
    all_tasks = []
    
    # 获取队列中的任务
    for level in range(scheduler.levels):
        for task in scheduler.queues[level]:
            if (status is None or task.status == status) and \
               (priority is None or task.priority == priority) and \
               (task_type is None or task.task_type == task_type):
                all_tasks.append(task.to_dict())
    
    # 获取运行中的任务
    for task_id, task in scheduler.running_tasks.items():
        if (status is None or task.status == status) and \
           (priority is None or task.priority == priority) and \
           (task_type is None or task.task_type == task_type):
            all_tasks.append(task.to_dict())
    
    # 获取历史任务
    for task in scheduler.task_history:
        if (status is None or task.status == status) and \
           (priority is None or task.priority == priority) and \
           (task_type is None or task.task_type == task_type):
            all_tasks.append(task.to_dict())
    
    # 按创建时间排序并限制数量
    all_tasks.sort(key=lambda x: x["created_at"], reverse=True)
    return all_tasks[:limit]

@app.get("/tasks/{task_id}", response_model=Dict[str, Any])
async def get_task(task_id: str = Path(..., description="任务ID")):
    """获取特定任务的详情"""
    task = await scheduler.get_task_by_id(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 未找到")
    
    return task.to_dict()

@app.get("/stats", response_model=Dict[str, Any])
async def get_stats():
    """获取调度器统计信息"""
    return await scheduler.get_stats()

@app.put("/scheduler/strategy", response_model=Dict[str, Any])
async def update_strategy(update: StrategyUpdate):
    """更新调度策略"""
    await scheduler.set_strategy(update.strategy)
    
    # 如果提供了时间片，更新时间片大小
    if update.time_slice is not None:
        scheduler.time_slice = update.time_slice
    
    return {
        "message": f"调度策略已更新为: {update.strategy}",
        "time_slice": scheduler.time_slice
    }

# 兼容旧版API
@app.post("/task")
async def submit_task_legacy(input: TaskInput):
    """提交新任务（兼容旧版API）"""
    return await submit_task(input)

# 启动服务器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
