# src/scheduler.py
import asyncio
from collections import deque, defaultdict
from typing import Optional, List, Dict
from enum import Enum
from .task import Task, TaskStatus

class SchedulingStrategy(str, Enum):
    """调度策略枚举"""
    PRIORITY_BASED = "priority_based"  # 基于优先级的调度
    ROUND_ROBIN = "round_robin"        # 时间片轮转
    PREEMPTIVE = "preemptive"          # 抢占式调度
    SHORTEST_JOB_FIRST = "shortest_job_first"  # 最短作业优先

class Scheduler:
    def __init__(self, levels=3, strategy=SchedulingStrategy.PRIORITY_BASED, time_slice=1.0):
        """
        初始化调度器
        
        Args:
            levels: 优先级队列数量
            strategy: 调度策略
            time_slice: 时间片大小(秒)
        """
        self.queues = defaultdict(deque)  # 多级队列：priority -> deque
        self.lock = asyncio.Lock()
        self.levels = levels
        self.strategy = strategy
        self.time_slice = time_slice
        self.current_task: Optional[Task] = None
        self.task_history: List[Task] = []
        self.running_tasks: Dict[str, Task] = {}  # 正在运行的任务 id -> task
        self.task_stats = {
            "total": 0,
            "completed": 0,
            "failed": 0,
            "avg_wait_time": 0,
            "avg_execution_time": 0
        }
    
    async def add_task(self, task: Task):
        """添加任务到队列"""
        async with self.lock:
            self.queues[task.priority].append(task)
            self.task_stats["total"] += 1
            # 如果是抢占式调度且新任务优先级高于当前任务，可以抢占
            if (self.strategy == SchedulingStrategy.PREEMPTIVE and 
                self.current_task and 
                task.priority < self.current_task.priority):
                # 标记当前任务需要被抢占
                self.current_task.preempted = True
    
    async def get_next_task(self) -> Optional[Task]:
        """根据调度策略获取下一个任务"""
        async with self.lock:
            if self.strategy == SchedulingStrategy.PRIORITY_BASED:
                return await self._get_priority_based_task()
            elif self.strategy == SchedulingStrategy.ROUND_ROBIN:
                return await self._get_round_robin_task()
            elif self.strategy == SchedulingStrategy.SHORTEST_JOB_FIRST:
                return await self._get_shortest_job_task()
            else:
                # 默认使用优先级调度
                return await self._get_priority_based_task()
    
    async def _get_priority_based_task(self) -> Optional[Task]:
        """基于优先级的调度策略"""
        for level in range(self.levels):
            if self.queues[level]:
                task = self.queues[level].popleft()
                self.current_task = task
                self.running_tasks[task.id] = task
                return task
        return None
    
    async def _get_round_robin_task(self) -> Optional[Task]:
        """时间片轮转调度策略"""
        # 从最高优先级开始，每个队列取一个任务
        for level in range(self.levels):
            if self.queues[level]:
                task = self.queues[level].popleft()
                # 设置时间片
                task.time_slice = self.time_slice
                self.current_task = task
                self.running_tasks[task.id] = task
                return task
        return None
    
    async def _get_shortest_job_task(self) -> Optional[Task]:
        """最短作业优先调度策略"""
        shortest_task = None
        shortest_time = float('inf')
        shortest_queue = None
        shortest_index = None
        
        # 查找所有队列中预估执行时间最短的任务
        for level in range(self.levels):
            for i, task in enumerate(self.queues[level]):
                if task.estimated_time < shortest_time:
                    shortest_time = task.estimated_time
                    shortest_task = task
                    shortest_queue = level
                    shortest_index = i
        
        # 如果找到了最短任务，从队列中移除并返回
        if shortest_task:
            self.queues[shortest_queue].remove(shortest_task)
            self.current_task = shortest_task
            self.running_tasks[shortest_task.id] = shortest_task
            return shortest_task
        
        return None
    
    async def task_completed(self, task: Task):
        """任务完成后的处理"""
        async with self.lock:
            if task.id in self.running_tasks:
                del self.running_tasks[task.id]
            
            # 更新任务历史和统计信息
            self.task_history.append(task)
            if task.status == TaskStatus.COMPLETED:
                self.task_stats["completed"] += 1
            elif task.status == TaskStatus.FAILED:
                self.task_stats["failed"] += 1
            
            # 计算平均等待和执行时间
            if task.wait_time:
                self.task_stats["avg_wait_time"] = (
                    (self.task_stats["avg_wait_time"] * (self.task_stats["completed"] + self.task_stats["failed"] - 1) + task.wait_time) / 
                    (self.task_stats["completed"] + self.task_stats["failed"])
                )
            
            if task.execution_time:
                self.task_stats["avg_execution_time"] = (
                    (self.task_stats["avg_execution_time"] * (self.task_stats["completed"] + self.task_stats["failed"] - 1) + task.execution_time) / 
                    (self.task_stats["completed"] + self.task_stats["failed"])
                )
    
    async def get_stats(self):
        """获取调度器统计信息"""
        async with self.lock:
            return {
                **self.task_stats,
                "queued_tasks": sum(len(q) for q in self.queues.values()),
                "running_tasks": len(self.running_tasks),
                "strategy": self.strategy
            }
    
    async def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """根据ID获取任务"""
        # 检查运行中的任务
        if task_id in self.running_tasks:
            return self.running_tasks[task_id]
        
        # 检查队列中的任务
        for level in range(self.levels):
            for task in self.queues[level]:
                if task.id == task_id:
                    return task
        
        # 检查历史任务
        for task in self.task_history:
            if task.id == task_id:
                return task
        
        return None
    
    async def set_strategy(self, strategy: SchedulingStrategy):
        """设置调度策略"""
        async with self.lock:
            self.strategy = strategy
            return {"message": f"调度策略已更新为: {strategy}"}
