import asyncio
import json
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

    def __init__(self, llm_service: LLMService, tools: List[Dict[str, Any]] = None, max_concurrent_tasks: int = 5):
        self.llm_service = llm_service
        self.agent = Agent(llm_service)
        self.planner_agent = PlannerAgent(llm_service)
        self.tools = tools if tools is not None else []
        self._tool_functions = {tool['function']['name']: tool['callable'] for tool in self.tools}
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

    async def shutdown(self):
        """Shuts down the scheduler gracefully."""
        if not self.is_running:
            return
        
        logger.info("Scheduler shutting down...")
        self.is_running = False
        
        # Cancel the main loop task
        if self._main_loop_task:
            self._main_loop_task.cancel()
            try:
                await self._main_loop_task
            except asyncio.CancelledError:
                logger.info("Main loop task cancelled.")
        
        # You might want to add logic here to handle any tasks still in queues
        # For now, we'll just log a warning if there are pending tasks.
        if not self.pending_queue.empty() or not self.resumption_queue.empty():
            logger.warning("Shutdown initiated with pending tasks still in queues.")

        logger.info("Scheduler shutdown complete.")

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
        return task

    def get_task_status(self, task_id: str) -> Optional[str]:
        """Gets the status of a task."""
        task = self.tasks.get(task_id)
        return task.status.value if task else None

    def get_task_result(self, task_id: str) -> Any:
        """Gets the result of a specific task."""
        task = self.tasks.get(task_id)
        if task:
            return task.result
        return None

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

                task_or_tuple = done.pop().result()
                if isinstance(task_or_tuple, tuple):
                    task, tool_result = task_or_tuple
                else:
                    task = task_or_tuple
                    tool_result = None

                logger.debug(f"Considering task {task.id} for execution. Available semaphore slots: {self.semaphore._value}")
                await self.semaphore.acquire()
                logger.debug(f"Semaphore acquired for task {task.id}. Remaining slots: {self.semaphore._value}")
                self.running_tasks_count += 1
                asyncio.create_task(self._drive_task(task, tool_result=tool_result))

            except asyncio.CancelledError:
                logger.info("Main loop cancelled.")
                pending_task_future.cancel()
                resumption_task_future.cancel()
                break
            except Exception as e:
                logger.error(f"An error occurred in the main loop: {e}", exc_info=True)

        logger.info("Scheduler main loop stopped.")

    async def _drive_task(self, task: Task, tool_result: Any = None):
        logger.info(f"--- Driving task: '{task.name}' ({task.id}), Type: {task.task_type.value}, Status: {task.status.value} ---")
        # This wrapper ensures the semaphore is acquired and released correctly.
        logger.debug(f"Task '{task.name}' ({task.id}) waiting for semaphore. Available: {self.semaphore._value}")
        await self.semaphore.acquire()
        logger.debug(f"Task '{task.name}' ({task.id}) acquired semaphore. Available: {self.semaphore._value}")
        try:

            # --- Task Decomposition Step ---
            if task.task_type == TaskType.PLANNING and task.status == TaskStatus.QUEUED:
                try:

                    plan = await self.planner_agent.decompose_task(task, tools=self.tools)
                    subtask_defs = plan.get('subtasks', []) if plan else []

                    if not subtask_defs:
                        # If planner returns nothing, mark parent as completed.
                        await self._handle_task_completion(task)
                        return

                    all_tasks_by_name = {t.name: t for t in self.tasks.values()}
                    newly_created_subtasks = {}

                    for sub_def in subtask_defs:
                        dep_ids = [all_tasks_by_name[dep_name].id for dep_name in sub_def.get('dependencies', []) if dep_name in all_tasks_by_name]
                        
                        subtask = Task(
                            name=sub_def['name'],
                            payload=sub_def['payload'],
                            task_type=TaskType(sub_def['task_type'].lower()),
                            parent_id=task.id,
                            dependencies=dep_ids
                        )
                        self.tasks[subtask.id] = subtask
                        task.waiting_for_subtasks.add(subtask.id)
                        newly_created_subtasks[sub_def['name']] = subtask

                    # Enqueue newly created tasks that are ready to run.
                    for subtask in newly_created_subtasks.values():
                        if subtask.is_ready():
                            await self.pending_queue.put(subtask)
                            logger.info(f"Enqueued new subtask '{subtask.name}' for parent {task.id}")

                    task.update_status(TaskStatus.WAITING_FOR_SUBTASKS)
                    logger.info(f"Task '{task.name}' ({task.id}) status is now WAITING_FOR_SUBTASKS.")

                except Exception as e:
                    logger.error(f"Failed to decompose task {task.id}: {e}", exc_info=True)
                    task.fail(f"Decomposition failed: {e}")
                    await self._handle_task_completion(task)
                finally:
                    # Release semaphore here for planning tasks, so subtasks can run.
                    logger.debug(f"Planning task '{task.name}' ({task.id}) releasing semaphore early.")
                    self.semaphore.release()
                    return

            # --- Standard Task Driving Step ---
            else:
                if task.id not in self.task_generators:
                    # This is the first time we're driving this task.
                    logger.info(f"Creating new generator for task '{task.name}' ({task.id}).")
                    generator = self.agent.process_task(task)
                    self.task_generators[task.id] = generator
                    send_value = None  # Start the generator for the first time
                else:
                    # Resuming a task that was waiting for a tool call.
                    logger.info(f"Resuming task '{task.name}' ({task.id}) with tool result.")
                    generator = self.task_generators[task.id]
                    send_value = tool_result

                task.update_status(TaskStatus.RUNNING)

                try:
                    logger.debug(f"Awaiting asend() for task {task.id}...")
                    tool_request = await generator.asend(send_value)
                    logger.debug(f"asend() completed for task {task.id}.")

                    if tool_request:
                        task.update_status(TaskStatus.WAITING_FOR_TOOL)
                        logger.info(f"Task '{task.name}' ({task.id}) is waiting for tool call.")
                        await self._execute_and_resume_task(task, tool_request)
                    else:
                        # Generator finished, task is complete.
                        task.complete(task.result)
                        await self._handle_task_completion(task)

                except StopAsyncIteration:
                    # Generator finished, task is complete.
                    logger.info(f"Task '{task.name}' ({task.id}) execution generator finished.")
                    task.complete(task.result)
                    await self._handle_task_completion(task)
                except Exception as e:
                    logger.error(f"An error occurred while driving task {task.id}: {e}", exc_info=True)
                    task.fail(str(e))
                    await self._handle_task_completion(task)
                finally:
                    # Release the semaphore if the task is fully finished or failed
                    if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                        self.semaphore.release()
                        logger.debug(f"Task '{task.name}' ({task.id}) released semaphore upon completion/failure.")

        except Exception as e:
            logger.error(f"Critical error in _drive_task for {task.id} ({task.name}): {e}", exc_info=True)
            task.fail(f"Scheduler-level error: {e}")
            await self._handle_task_completion(task)
        finally:
            logger.debug(f"Task '{task.name}' ({task.id}) releasing semaphore. Available: {self.semaphore._value + 1}")
            self.semaphore.release()

    async def _execute_and_resume_task(self, task: Task, tool_calls: list):
        """Executes tool calls and places the task in the resumption queue."""
        logger.info(f"Executing {len(tool_calls.get('calls', []))} tool calls for task {task.id}...")
        tool_results = []
        for tool_call in tool_calls.get("calls", []):
            tool_name = tool_call.function.name
            if tool_name in self._tool_functions:
                try:
                    # Arguments are a JSON string, so we need to parse them
                    args = json.loads(tool_call.function.arguments)
                    logger.info(f"Calling tool `{tool_name}` with args: {args}")
                    # In a real-world scenario, you might need to handle async tool functions
                    result = self._tool_functions[tool_name](**args)
                    content = json.dumps({"result": result})
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name} for task {task.id}: {e}", exc_info=True)
                    content = json.dumps({"error": str(e)})
            else:
                logger.warning(f"Tool `{tool_name}` not found for task {task.id}.")
                content = json.dumps({"error": f"Tool '{tool_name}' not found."})

            tool_results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_name,
                "content": content,
            })

        task.result = tool_results
        logger.info(f"Tool calls for task {task.id} completed. Adding to resumption queue.")
        await self.resumption_queue.put((task, tool_results))

    async def _handle_task_completion(self, task: Task):
        """Handles the logic for when a task finishes, either by completing or failing."""
        async with self._lock:
            if task.status == TaskStatus.COMPLETED:
                self.completed_tasks_count += 1
                logger.info(f"--- Task '{task.name}' ({task.id}) COMPLETED ---")
            elif task.status == TaskStatus.FAILED:
                self.failed_tasks_count += 1
                logger.error(f"--- Task '{task.name}' ({task.id}) FAILED: {task.result} ---")

            # Clean up generator for the completed/failed task
            if task.id in self.task_generators:
                del self.task_generators[task.id]

            # --- Dependency Resolution for other tasks ---
            # Find all tasks that were waiting for this one to finish
            dependent_tasks = [t for t in self.tasks.values() if task.id in t.waiting_for_dependencies]
            for dep_task in dependent_tasks:
                dep_task.waiting_for_dependencies.remove(task.id)
                logger.info(f"Resolved dependency '{task.name}' for '{dep_task.name}'.")
                if dep_task.is_ready():
                    logger.info(f"Task '{dep_task.name}' is now ready. Enqueuing.")
                    await self.pending_queue.put(dep_task)

            # --- Parent Task Completion ---
            # If this was a subtask, check if its parent is now finished
            if task.parent_id and task.parent_id in self.tasks:
                parent_task = self.tasks[task.parent_id]
                if task.id in parent_task.waiting_for_subtasks:
                    parent_task.waiting_for_subtasks.remove(task.id)
                
                # If the parent has no more subtasks to wait for, it's complete
                if not parent_task.waiting_for_subtasks:
                    logger.info(f"All subtasks for parent '{parent_task.name}' are complete.")
                    # The result of the parent is the result of its final subtask.
                    # We assume the last completed subtask holds the final result.
                    parent_task.complete(result=task.result)
                    # Recursively handle the completion of the parent task
                    await self._handle_task_completion(parent_task)

                    # Also, if this parent task itself has a parent, remove it from the grandparent's waiting list
                    if parent_task.parent_id and parent_task.parent_id in self.tasks:
                        grandparent_task = self.tasks[parent_task.parent_id]
                        if parent_task.id in grandparent_task.waiting_for_subtasks:
                            grandparent_task.waiting_for_subtasks.remove(parent_task.id)

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
