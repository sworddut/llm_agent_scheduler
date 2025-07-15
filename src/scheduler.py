import asyncio
import logging
from typing import Dict, Any, Optional, AsyncGenerator, List

from .task import Task, TaskStatus, TaskType
from .agent import Agent, PlannerAgent
from .llm_service import LLMService

logger = logging.getLogger(__name__)

class Scheduler:
    """
    An OS-like scheduler that manages the lifecycle of LLM agent tasks.
    It drives tasks, handles I/O (tool calls) by pausing and resuming tasks,
    and enables true concurrent execution.
    """

    def __init__(self, llm_service: LLMService, max_concurrent_tasks: int = 5):
        self.llm_service = llm_service
        self.agent = Agent(llm_service)
        self.planner_agent = PlannerAgent(llm_service)
        self.max_concurrent_tasks = max_concurrent_tasks
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        self.pending_queue = asyncio.Queue()
        self.resumption_queue = asyncio.Queue()
        
        self.tasks: Dict[str, Task] = {}
        self.task_generators: Dict[str, AsyncGenerator] = {}
        
        self.is_running = False
        self._main_loop_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Statistics
        self.completed_tasks_count = 0
        self.failed_tasks_count = 0
        self.running_tasks_count = 0
        self.pending_tasks_count = 0
        self.waiting_tasks_count = 0

    async def start(self):
        """Starts the scheduler's main loop in the background."""
        if self.is_running:
            logger.warning("Scheduler is already running.")
            return
        self.is_running = True
        self._main_loop_task = asyncio.create_task(self._main_loop())
        logger.info("Scheduler started.")

    async def stop(self):
        """Stops the scheduler's main loop gracefully."""
        self.is_running = False
        if self._main_loop_task:
            self._main_loop_task.cancel()
            try:
                await self._main_loop_task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped.")

    async def add_task(self, task: Task):
        """Adds a new task to the scheduler's queue."""
        async with self._lock:
            self.tasks[task.id] = task
            self.pending_tasks_count += 1
        await self.pending_queue.put(task)
        logger.info(f"Task {task.id} added to the queue.")

    async def _main_loop(self):
        """The core loop that sources tasks from pending and resumption queues."""
        logger.info("Scheduler main loop started.")
        while self.is_running:
            pending_task_future = asyncio.create_task(self.pending_queue.get())
            resumption_task_future = asyncio.create_task(self.resumption_queue.get())

            try:
                done, pending = await asyncio.wait(
                    [pending_task_future, resumption_task_future],
                    return_when=asyncio.FIRST_COMPLETED
                )

                for future in pending:
                    future.cancel()

                task = done.pop().result()
                await self.semaphore.acquire()
                asyncio.create_task(self._drive_task(task))

            except asyncio.CancelledError:
                logger.info("Main loop cancelled.")
                pending_task_future.cancel()
                resumption_task_future.cancel()
                break
            except Exception as e:
                logger.error(f"An error occurred in the main loop: {e}", exc_info=True)

        logger.info("Scheduler main loop stopped.")

    async def _drive_task(self, task: Task, tool_result: Any = None):
        # This wrapper ensures the semaphore is acquired and released correctly.
        await self.semaphore.acquire()
        try:

            # --- Task Decomposition Step ---
            if task.task_type == TaskType.PLANNING and task.status == TaskStatus.QUEUED:
                try:

                    plan = await self.planner_agent.decompose_task(task)
                    subtask_defs = plan.get('subtasks', []) if plan else []

                    if not subtask_defs:
                        # If planner returns nothing, mark parent as completed.
                        await self._handle_task_completion(task)
                        return

                    subtask_map = {}
                    for sub_def in subtask_defs:

                        subtask = Task(
                            name=sub_def['name'],
                            payload=sub_def['payload'],
                            task_type=TaskType(sub_def['task_type'].lower()),
                            parent_id=task.id,
                            dependencies=[subtask_map[dep_name].id for dep_name in sub_def.get('dependencies', [])]
                        )
                        self.tasks[subtask.id] = subtask
                        task.subtasks.append(subtask.id)
                        subtask_map[sub_def['name']] = subtask


                    # Enqueue subtasks with no dependencies
                    for subtask in subtask_map.values():
                        if not subtask.dependencies:
                            await self.pending_queue.put(subtask)
                            logger.info(f"Enqueued subtask '{subtask.name}' for parent {task.id}")

                    task.status = TaskStatus.WAITING_FOR_SUBTASKS
                    logger.info(f"Parent task {task.id} is now WAITING_FOR_SUBTASKS.")

                except Exception as e:
                    logger.error(f"Failed to decompose task {task.id}: {e}", exc_info=True)
                    task.fail(f"Decomposition failed: {e}")
                    await self._handle_task_completion(task)
                return

            # --- Standard Task Driving Step ---
            if task.id not in self.task_generators:
                if task.status == TaskStatus.QUEUED:
                    task.start()
                    self.task_generators[task.id] = self.agent.process_task(task)
                    send_value = None
                else:
                    logger.warning(f"Task {task.id} (status: {task.status}) has no generator. Re-creating.")
                    self.task_generators[task.id] = self.agent.process_task(task)
                    send_value = tool_result
            else:
                task.status = TaskStatus.RUNNING
                send_value = tool_result

            generator = self.task_generators[task.id]
            try:
                tool_request = await generator.asend(send_value)
                if tool_request and tool_request.get("type") == "tool_call":
                    task.status = TaskStatus.WAITING_FOR_TOOL
                    logger.info(f"Task {task.id} ({task.name}) is waiting for tool call.")
                    asyncio.create_task(self._execute_and_resume_task(task, tool_request))

            except StopAsyncIteration:
                logger.info(f"Task {task.id} ({task.name}) execution generator finished.")
                await self._handle_task_completion(task)

            except Exception as e:
                logger.error(f"Generator for task {task.id} ({task.name}) failed: {e}", exc_info=True)
                task.fail(str(e))
                await self._handle_task_completion(task)

        except Exception as e:
            logger.error(f"Critical error in _drive_task for {task.id} ({task.name}): {e}", exc_info=True)
            task.fail(f"Scheduler-level error: {e}")
            await self._handle_task_completion(task)
        finally:
            self.semaphore.release()

    async def _execute_and_resume_task(self, task: Task, tool_calls: list):
        """Executes tool calls and places the task in the resumption queue."""
        logger.info(f"Executing {len(tool_calls)} tool calls for task {task.id}...")
        await asyncio.sleep(2)  # Simulate I/O
        import json
        tool_results = []
        for tool_call in tool_calls.get("tool_calls", []):
            # In a real scenario, you'd execute the tool and get a real result.
            tool_results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": json.dumps({"temperature": "30", "unit": "celsius"})  # Mocked
            })

        task.result = tool_results
        logger.info(f"Tool calls for task {task.id} completed. Adding to resumption queue.")
        await self.resumption_queue.put(task)

    async def _handle_task_completion(self, task: Task):
        """Handles the completion logic for a task, including dependency resolution."""


        if task.status == TaskStatus.COMPLETED:
            self.completed_tasks_count += 1
        elif task.status == TaskStatus.FAILED:
            self.failed_tasks_count += 1

        if task.id in self.task_generators:
            del self.task_generators[task.id]

        if task.parent_id:
            parent_task = self.tasks.get(task.parent_id)
            if parent_task and parent_task.status == TaskStatus.WAITING_FOR_SUBTASKS:
                all_subtasks_finished = all(
                    self.tasks.get(sub_id).status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
                    for sub_id in parent_task.subtasks
                )
    
                if all_subtasks_finished:
                    logger.info(f"All subtasks for parent {parent_task.id} are finished.")
                    if any(self.tasks.get(sub_id).status == TaskStatus.FAILED for sub_id in parent_task.subtasks):
                        parent_task.fail("One or more subtasks failed.")
                    else:
                        parent_task.complete("All subtasks completed successfully.")
                    await self._handle_task_completion(parent_task)

        for other_task in self.tasks.values():
            if task.id in other_task.dependencies:
                other_task.dependencies.remove(task.id)

                if not other_task.dependencies:
                    logger.info(f"All dependencies for task {other_task.id} are resolved. Enqueuing.")
                    await self.pending_queue.put(other_task)

    async def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Retrieves a task by its ID."""
        return self.tasks.get(task_id)

    async def get_stats(self) -> Dict[str, Any]:
        """Returns current statistics about the scheduler."""
        async with self._lock:
            running_tasks = self.max_concurrent_tasks - self.semaphore._value
            return {
                "is_running": self.is_running,
                "running_tasks": running_tasks,
                "pending_tasks": self.pending_queue.qsize(),
                "resumption_queue_size": self.resumption_queue.qsize(),
                "total_known_tasks": len(self.tasks),
                "completed_tasks": self.completed_tasks_count,
                "failed_tasks": self.failed_tasks_count,
                "max_concurrent_tasks": self.max_concurrent_tasks
            }
