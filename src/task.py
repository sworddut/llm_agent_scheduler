# src/task.py
from enum import Enum
from uuid import uuid4
from typing import Union, Optional, Dict, Any, List
import time
from datetime import datetime

class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PREEMPTED = "preempted"  # 新增：被抢占状态

class TaskType(str, Enum):
    """任务类型枚举"""
    FUNCTION_CALL = "function_call"  # 函数调用
    API_REQUEST = "api_request"      # API请求
    FILE_OPERATION = "file_operation"  # 文件操作
    CUSTOM = "custom"                # 自定义任务

class Task:
    def __init__(self, name: str, payload: dict, priority: int = 1, 
                 task_type: Union[TaskType, str] = TaskType.FUNCTION_CALL,
                 estimated_time: float = 1.0,
                 tags: Optional[List[str]] = None):
        self.id = str(uuid4())
        self.name = name
        self.payload = payload
        self.priority = priority  # 0: high, 1: normal, 2: low
        self.status = TaskStatus.QUEUED
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        
        # 任务类型处理
        if isinstance(task_type, str):
            # 尝试将字符串转换为 TaskType 枚举
            try:
                self.task_type = TaskType(task_type.lower())
            except ValueError:
                # 如果不是预定义的类型，使用 CUSTOM 类型，并保存原始字符串
                self.task_type = TaskType.CUSTOM
                self.metadata = {"custom_task_type": task_type}
        else:
            self.task_type = task_type
            
        # 标签
        self.tags = tags or []
        
        # 时间跟踪
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.wait_time: Optional[float] = None  # 等待时间（秒）
        self.execution_time: Optional[float] = None  # 执行时间（秒）
        
        # 调度相关
        self.estimated_time = estimated_time  # 预估执行时间（秒）
        self.time_slice: Optional[float] = None  # 分配的时间片（秒）
        self.preempted = False  # 是否被抢占
        
        # 依赖和子任务
        self.dependencies: List[str] = []  # 依赖任务ID列表
        self.subtasks: List[str] = []  # 子任务ID列表
        
        # 元数据
        self.metadata: Dict[str, Any] = {}  # 存储任务相关的元数据
        self.tags: List[str] = []  # 任务标签

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
        # 处理任务类型，如果是自定义类型，返回原始字符串
        task_type = self.task_type
        if hasattr(self, 'metadata') and 'custom_task_type' in self.metadata:
            task_type = self.metadata['custom_task_type']
            
        return {
            "id": self.id,
            "name": self.name,
            "priority": self.priority,
            "status": self.status.value if hasattr(self.status, 'value') else self.status,
            "task_type": task_type.value if hasattr(task_type, 'value') else task_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "wait_time": self.wait_time,
            "execution_time": self.execution_time,
            "estimated_time": self.estimated_time,
            "result": self.result,
            "error": self.error,
            "tags": self.tags,
            "dependencies": self.dependencies,
            "subtasks": self.subtasks,
            "metadata": getattr(self, 'metadata', {})
        }

    def __repr__(self):
        return f"<Task {self.name}({self.id[:6]}) p={self.priority} status={self.status}>"
