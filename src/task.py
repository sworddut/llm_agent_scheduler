# src/task.py
from enum import Enum
from uuid import uuid4
from typing import Union, Optional, Dict, Any, List
import time
from datetime import datetime

class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_FOR_TOOL = "waiting_for_tool"
    WAITING_FOR_SUBTASKS = "waiting_for_subtasks"  # 新增状态：等待工具调用完成
    COMPLETED = "completed"
    FAILED = "failed"
    PREEMPTED = "preempted"  # 新增：被抢占状态

class TaskType(str, Enum):
    """任务类型枚举"""
    FUNCTION_CALL = "function_call"  # 一个直接的工具或函数调用
    REASONING = "reasoning"          # 需要LLM进行推理
    PLANNING = "planning"            # 一个需要被分解的复杂任务

class Task:
    def __init__(self, name: str, payload: dict, priority: int = 1, 
                 task_type: Union[TaskType, str] = TaskType.FUNCTION_CALL,
                 estimated_time: float = 1.0,
                 tags: Optional[List[str]] = None,
                 parent_id: Optional[str] = None,
                 dependencies: Optional[List[str]] = None,
                 is_decomposable: bool = False):

        self.id = str(uuid4())
        self.name = name
        self.payload = payload
        self.priority = priority
        self.task_type = task_type if isinstance(task_type, TaskType) else TaskType(task_type)
        self.estimated_time = estimated_time
        self.tags = tags if tags is not None else []

        # --- Dependency and hierarchy fields ---
        self.parent_id = parent_id
        self.dependencies = dependencies if dependencies is not None else []
        self.is_decomposable = is_decomposable
        self.subtasks: List[str] = []  # Populated by the scheduler for parent tasks

        # --- State and Result ---
        self.status = TaskStatus.QUEUED
        self.result: Optional[Any] = None
        self.error: Optional[str] = None

        # --- Time Tracking ---
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.wait_time: Optional[float] = None
        self.execution_time: Optional[float] = None

        # --- Scheduler Internals ---
        self.preempted = False
        self.metadata: Dict[str, Any] = {}

    def start(self):
        """开始执行任务"""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()
        self.wait_time = (self.started_at - self.created_at).total_seconds()
        return self
    
    def complete(self, result: Any = None):
        """完成任务"""
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now()
        if self.started_at:
            self.execution_time = (self.completed_at - self.started_at).total_seconds()
        return self
    
    def fail(self, error: str = ""):
        """任务失败"""
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = datetime.now()
        if self.started_at:
            self.execution_time = (self.completed_at - self.started_at).total_seconds()
        return self
    
    def preempt(self):
        """任务被抢占"""
        self.status = TaskStatus.PREEMPTED
        self.preempted = True
        return self
    
    def add_dependency(self, task_id: str):
        """添加依赖任务"""
        if task_id not in self.dependencies:
            self.dependencies.append(task_id)
        return self
    
    def add_subtask(self, task_id: str):
        """添加子任务"""
        if task_id not in self.subtasks:
            self.subtasks.append(task_id)
        return self
    
    def add_tag(self, tag: str):
        """添加标签"""
        if tag not in self.tags:
            self.tags.append(tag)
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        return {
            "id": self.id,
            "name": self.name,
            "priority": self.priority,
            "status": self.status.value,
            "task_type": self.task_type.value,
            "is_decomposable": self.is_decomposable,

            "parent_id": self.parent_id,
            "dependencies": self.dependencies,
            "subtasks": self.subtasks,

            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "wait_time": self.wait_time,
            "execution_time": self.execution_time,

            "result": self.result,
            "error": self.error,
            "tags": self.tags,
            "metadata": self.metadata
        }

    def __repr__(self):
        return f"<Task {self.name}({self.id[:6]}) p={self.priority} status={self.status}>"
