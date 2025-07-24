# src/task.py
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

class TaskStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_FOR_TOOL = "waiting_for_tool"
    WAITING_FOR_SUBTASKS = "waiting_for_subtasks"
    COMPLETED = "completed"
    FAILED = "failed"
    PREEMPTED = "preempted"

class TaskType(Enum):
    PLANNING = "planning"
    TOOL_CALL = "tool_call"
    FINAL_SUMMARY = "final_summary"

class Task:
    def __init__(self, name: str, payload: Dict[str, Any], task_type: TaskType, dependencies: Optional[List[str]] = None):
        self.id = str(uuid.uuid4())
        self.name = name
        self.payload = payload
        self.task_type = task_type
        self.parent: Optional['Task'] = None # Will be set by the scheduler
        self.dependencies_names = dependencies or [] # Store dependency names temporarily
        
        # --- Refactored State Management with Direct References ---
        # Dependencies that this task is waiting for.
        self.waiting_for_dependencies: Set['Task'] = set()
        # Subtasks that this task has spawned and is waiting for.
        self.waiting_for_subtasks: Set['Task'] = set()
        
        self.status = TaskStatus.QUEUED
        self.result: Optional[Any] = None
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def __repr__(self):
        return f"Task(id={self.id}, name='{self.name}', type={self.task_type.name}, status={self.status.name})"

    def is_ready(self) -> bool:
        """A task is ready to run if it's not waiting on any dependencies."""
        return not self.waiting_for_dependencies

    def is_complete(self) -> bool:
        """A task is considered fully complete if its subtasks are all done."""
        return not self.waiting_for_subtasks

    def update_status(self, status: TaskStatus):
        self.status = status
        self.updated_at = datetime.now()

    def complete(self, result: Any):
        self.update_status(TaskStatus.COMPLETED)
        self.result = result

    def fail(self, error_message: str):
        self.update_status(TaskStatus.FAILED)
        self.result = {"error": error_message}
