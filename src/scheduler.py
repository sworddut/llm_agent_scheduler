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

    def __init__(self, llm_service: LLMService, max_concurrent_tasks: int = 5):
        self.llm_service = llm_service
        self.agent = Agent(llm_service)
        self.planner_agent = PlannerAgent(llm_service)
        self.max_concurrent_tasks = max_concurrent_tasks
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        self.pending_queue = asyncio.Queue()
        self.resumption_queue = asyncio.Queue()
        
        self.tasks: Dict[str, Task] = {}
        self.task_name_to_id: Dict[str, str] = {}
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

    @classmethod
    async def create(cls, llm_service: LLMService, max_concurrent_tasks: int = 5):
        """Asynchronously creates and initializes a Scheduler instance."""
        return cls(llm_service, max_concurrent_tasks)



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
            self.task_name_to_id[task.name] = task.id
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
                    plan = await self.planner_agent.decompose_task(task)
                    if plan:
                        await self._decompose_and_enqueue_subtasks(task, plan)
                        task.update_status(TaskStatus.WAITING_FOR_SUBTASKS)
                    else:
                        task.complete("Planner returned no plan.")
                        await self._handle_task_completion(task)
                except Exception as e:
                    logger.error(f"Failed to decompose task {task.id}: {e}", exc_info=True)
                    task.fail(f"Decomposition failed: {e}")
                    await self._handle_task_completion(task)
                finally:
                    self.semaphore.release()
                    return

            # --- Final Summary Task Preparation ---
            if task.task_type == TaskType.FINAL_SUMMARY and task.status == TaskStatus.QUEUED:
                summary_prompt = "Synthesize the results from the previous steps to provide a final answer to the user's request.\n\n"
                summary_prompt += f"Original user request: {task.parent.payload.get('goal', 'N/A')}\n\n"
                summary_prompt += "Here are the results from the executed tools:\n"
                
                for dep_task in task.waiting_for_dependencies:
                    if dep_task:
                        summary_prompt += f"- Result from '{dep_task.name}':\n"
                        summary_prompt += f"  - Status: {dep_task.status.value}\n"
                        summary_prompt += f"  - Result: {json.dumps(dep_task.result, indent=2)}\n\n"
                print('=='*20)
                print(summary_prompt)
         
                print('=='*20)
                task.payload['prompt'] = summary_prompt
                logger.info(f"Prepared context for FINAL_SUMMARY task '{task.name}'.")

            # --- Standard Task Driving Step ---
            else:
                # For FINAL_SUMMARY, we also create a generator, but after its context has been prepared.
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
                        # The task will be added to the resumption_queue by _execute_and_resume_task
                        asyncio.create_task(self._execute_and_resume_task(task, tool_request))
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

    async def _decompose_and_enqueue_subtasks(self, parent_task: Task, plan: Dict[str, Any]):
        subtask_defs = plan.get('subtasks', [])
        if not subtask_defs:
            logger.warning(f"Task {parent_task.id} produced a plan with no subtasks.")
            return

        # First pass: create all task objects
        new_tasks_map = {}
        for subtask_def in subtask_defs:
            task_type = TaskType[subtask_def.get('task_type', 'SIMPLE').upper()]
            new_task = Task(
                name=subtask_def['name'],
                payload=subtask_def.get('payload', {}),
                task_type=task_type,
                dependencies=subtask_def.get('dependencies', [])
            )
            new_tasks_map[new_task.name] = new_task

        # Second pass: resolve dependencies and parentage
        for task_name, task in new_tasks_map.items():
            task.parent = parent_task
            parent_task.waiting_for_subtasks.add(task)

            for dep_name in task.dependencies_names:
                dep_task = new_tasks_map.get(dep_name)

                if dep_task:
                    task.waiting_for_dependencies.add(dep_task)
                else:
                    logger.warning(f"Could not find dependency '{dep_name}' for task '{task.name}' among subtasks.")
            
            # Add to scheduler's main task dict and queue
            await self.add_task(task)

        for subtask in new_tasks_map.values():
            # Only queue tasks that have no dependencies initially.
            # Tasks with dependencies will be queued by _handle_task_completion once they are met.
            if not subtask.waiting_for_dependencies:
                await self.pending_queue.put(subtask)
                logger.info(f"Enqueued subtask '{subtask.name}' for parent '{parent_task.name}'.")
            else:
                logger.info(f"Subtask '{subtask.name}' has dependencies, deferring queueing.")

    async def _execute_and_resume_task(self, task: Task, tool_request: dict):
        """Executes tool calls using LLMService and places the task in the resumption queue."""
        logger.info(f"Executing tool calls for task {task.id} via LLMService...")
        tool_calls = tool_request.get("calls", [])
        tool_results = []

        # This can be extended to handle multiple parallel tool calls in the future
        for tool_call in tool_calls:
            # Delegate the execution process to the LLMService
            result_content = await self.llm_service.process_and_call_tool(tool_call)

            # Format the result into the list structure the agent expects
            tool_results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": result_content,  # The direct output from the service
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
            dependent_tasks = [t for t in self.tasks.values() if task in t.waiting_for_dependencies]
            for dep_task in dependent_tasks:
                dep_task.waiting_for_dependencies.remove(task)
                logger.info(f"Resolved dependency '{task.name}' for '{dep_task.name}'. Remaining: {[d.name for d in dep_task.waiting_for_dependencies]}")
                if not dep_task.waiting_for_dependencies:
                    logger.info(f"All dependencies for '{dep_task.name}' are complete. Re-queueing.")
                    await self._enqueue_task(dep_task)

            # --- Parent Task Completion ---
            # If this was a subtask, check if its parent is now finished
            parent_task = task.parent
            if parent_task and task in parent_task.waiting_for_subtasks:
                parent_task.waiting_for_subtasks.remove(task)
                
                if not parent_task.waiting_for_subtasks:
                    logger.info(f"All subtasks for task '{parent_task.name}' are complete.")
                    if parent_task.task_type == TaskType.PLANNING:
                        parent_task.complete(result=task.result)
                    else:
                        # If a regular task group finishes, we might need to re-queue the parent
                        # This case might need more sophisticated handling depending on desired logic
                        pass

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
