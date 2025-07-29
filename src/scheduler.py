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
        
        self.tasks: Dict[str, Task] = {}
        self.task_name_to_id: Dict[str, str] = {}
        self.task_generators: Dict[str, AsyncGenerator] = {}
        
        self.is_running = False
        self.active_tasks: Set[asyncio.Task] = set()
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



    async def start(self):
        """Starts the scheduler."""
        if self.is_running:
            logger.warning("Scheduler is already running.")
            return
        self.is_running = True
        logger.info("Scheduler started.")

    async def shutdown(self):
        """Shuts down the scheduler gracefully, cancelling all active tasks."""
        if not self.is_running:
            return
        
        logger.info("Scheduler shutting down...")
        self.is_running = False
        
        # Cancel all active tasks
        for task in self.active_tasks:
            task.cancel()
        
        await asyncio.gather(*self.active_tasks, return_exceptions=True)
        logger.info("Scheduler shutdown complete.")

    async def add_task(self, task: Task) -> Task:
        """Adds a new root task and starts its execution in the background."""
        if not self.is_running:
            await self.start()

        async with self._lock:
            self.tasks[task.id] = task
            self.task_name_to_id[task.name] = task.id
        
        # Create a background task for the recursive execution
        exec_task = asyncio.create_task(self._drive_task(task))
        self.active_tasks.add(exec_task)
        # Remove the task from the set upon completion to avoid memory leaks
        exec_task.add_done_callback(self.active_tasks.discard)

        logger.info(f"Task '{task.name}' ({task.id}) execution started in the background.")
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



    async def _drive_task(self, task: Task, tool_result: Any = None):
        """The entry point for task execution, now simplified to call the recursive executor."""
        logger.info(f"--- Starting execution for root task: '{task.name}' ({task.id}) ---")
        try:
            await self.execute_task_recursively(task)
        except Exception as e:
            logger.error(f"A top-level error occurred while driving task {task.id}: {e}", exc_info=True)
            task.fail(f"Execution failed: {e}")
        finally:
            logger.info(f"--- Finished execution for root task: '{task.name}' ({task.id}), Status: {task.status.value} ---")

    async def execute_task_recursively(self, task: Task):
        """Executes a task and its subtasks in a recursive, DFS-like manner."""
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            logger.warning(f"Skipping already completed/failed task: {task.name}")
            return

        logger.info(f"Executing node: '{task.name}' ({task.task_type.value}), Status: {task.status.value}")
        task.update_status(TaskStatus.RUNNING)

        try:
            # --- 1. Decompose Task if it's a PLANNING task ---
            if task.task_type == TaskType.PLANNING:
                # Decompose the task into subtasks
                subtask_payloads = await self._drive_task_for_subtasks(task)
                if subtask_payloads:
                    await self._create_and_link_subtasks(task, subtask_payloads)

                    # Separate summary task from other subtasks
                    summary_task = None
                    tool_subtasks = []
                    for subtask in task.subtasks:
                        if subtask.task_type == TaskType.FINAL_SUMMARY:
                            summary_task = subtask
                        else:
                            tool_subtasks.append(subtask)

                    # 1. Execute all tool-related subtasks concurrently and wait for them to complete
                    if tool_subtasks:
                        logger.info(f"Executing {len(tool_subtasks)} tool subtasks for '{task.name}'.")
                        subtask_executions = [self.execute_task_recursively(st) for st in tool_subtasks]
                        await asyncio.gather(*subtask_executions)
                    
                    # 2. Once all tool subtasks are done, execute the final summary task
                    if summary_task:
                        logger.info(f"All tool subtasks for '{task.name}' are complete. Executing summary task.")
                        await self.execute_task_recursively(summary_task)

            else: # Leaf task (TOOL_CALL or FINAL_SUMMARY)
                await self._execute_leaf_task(task)

            # --- 3. Finalize task completion after all sub-work is done ---
            if task.is_complete():
                task.complete(task.result) # Result should be set by leaf nodes or aggregated
                logger.info(f"Task '{task.name}' and all its subtasks are complete.")
            else:
                # This case might indicate an issue if not all subtasks completed
                logger.warning(f"Task '{task.name}' finished execution but is not marked as complete.")

        except Exception as e:
            logger.error(f"Error executing task '{task.name}': {e}", exc_info=True)
            task.fail(str(e))
        finally:
            await self._handle_task_completion(task)

    async def _drive_task_for_subtasks(self, task: Task) -> list[dict]:
        """Drives the planner agent to decompose a task and returns the subtask definitions."""
        logger.info(f"Driving planner for task '{task.name}'...")
        
        # The planner agent returns a plan with subtasks
        plan = await self.planner_agent.decompose_task(task)
        subtask_payloads = plan.get('subtasks', [])
        
        if not subtask_payloads:
            logger.info(f"Planner returned no subtasks for '{task.name}'.")
            return []
            
        logger.info(f"Planner for '{task.name}' returned {len(subtask_payloads)} subtasks.")
        return subtask_payloads

    async def _execute_leaf_task(self, task: Task):
        """Handles the execution of a single, non-decomposable task."""
        logger.info(f"Executing leaf task: '{task.name}'")

        # Prepare context for FINAL_SUMMARY tasks
        if task.task_type == TaskType.FINAL_SUMMARY:
            if not all(dep.status == TaskStatus.COMPLETED for dep in task.waiting_for_dependencies):
                logger.warning(f"Deferring FINAL_SUMMARY '{task.name}' as dependencies are not met.")
                return
            task.payload['prompt'] = self._prepare_summary_prompt(task)

        # Drive the agent to get a tool request or final answer
        generator = self.agent.process_task(task)
        self.task_generators[task.id] = generator

        try:
            try:
                # Loop to drive the agent generator.
                next_val = await anext(generator)
                while True:
                    if next_val:
                        # If the agent returns a tool request, execute it.
                        tool_results = await self._execute_tool_calls(next_val)
                        # Send the tool results back to the agent to continue.
                        next_val = await generator.asend(tool_results)
                    else:
                        # If the agent yields None, it's waiting. Continue driving.
                        next_val = await anext(generator)
            except StopAsyncIteration as e:
                # The generator has finished.
                # For FINAL_SUMMARY, the return value is the summary.
                if task.task_type == TaskType.FINAL_SUMMARY:
                    summary = getattr(e, 'value', None)
                    task.result = {"summary": summary}
                    logger.info(f"Final summary generated for task '{task.name}': {summary}")
                # For other tasks (TOOL_CALL), the result is set via side effects
                # within the agent (e.g., storing tool output), so no special
                # handling is needed here for the return value.

        except Exception as e:
            logger.error(f"An error occurred while executing task '{task.name}': {e}", exc_info=True)
            task.fail(str(e))

        logger.info(f"Leaf task '{task.name}' finished processing.")

    async def _create_and_link_subtasks(self, parent_task: Task, subtask_defs: List[Dict[str, Any]]) -> None:
        """Creates task objects from definitions and links them in the task tree."""
        new_tasks_map = {}
        for subtask_def in subtask_defs:
            task_type_str = subtask_def.get('task_type', 'TOOL_CALL').upper()
            task_type = TaskType[task_type_str]
            new_task = Task(
                name=subtask_def['name'],
                payload=subtask_def.get('payload', {}),
                task_type=task_type,
                dependencies=subtask_def.get('dependencies', [])
            )
            new_tasks_map[new_task.name] = new_task
            self.tasks[new_task.id] = new_task # Register task globally

        for task in new_tasks_map.values():
            task.parent = parent_task
            parent_task.subtasks.add(task)
            for dep_name in task.dependencies_names:
                if dep_task := new_tasks_map.get(dep_name):
                    task.waiting_for_dependencies.add(dep_task)
                else:
                    logger.warning(f"Dependency '{dep_name}' for task '{task.name}' not found in current plan.")

    def _prepare_summary_prompt(self, task: Task) -> str:
        """Prepares the prompt for a FINAL_SUMMARY task."""
        summary_prompt = "Synthesize the results from the previous steps to provide a final answer.\n\n"
        if task.parent:
             summary_prompt += f"Original user request: {task.parent.payload.get('goal', 'N/A')}\n\n"
        summary_prompt += "Here are the results from the executed subtasks:\n"
        for dep_task in task.waiting_for_dependencies:
            summary_prompt += f"- Result from '{dep_task.name}': {json.dumps(dep_task.result, indent=2)}\n"
        return summary_prompt

    async def _execute_tool_calls(self, tool_request: dict) -> List[Dict[str, Any]]:
        """Executes tool calls and returns the results directly."""
        logger.info(f"Executing tool calls: {tool_request.get('calls')}")
        tool_calls = tool_request.get("calls", [])
        tool_results = []
        
        async def execute_single_call(tool_call):
            result_content = await self.llm_service.process_and_call_tool(tool_call)
            return {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": result_content,
            }

        # Execute all tool calls concurrently
        results = await asyncio.gather(*[execute_single_call(tc) for tc in tool_calls])
        logger.info(f"Tool calls completed with results.")
        return results

    async def _handle_task_completion(self, task: Task):
        """Handles the final logic for when a task finishes, primarily for cleanup and logging."""
        async with self._lock:
            if task.status == TaskStatus.COMPLETED:
                self.completed_tasks_count += 1
                logger.info(f"--- Task '{task.name}' ({task.id}) finalized as COMPLETED ---")
            elif task.status == TaskStatus.FAILED:
                self.failed_tasks_count += 1
                logger.error(f"--- Task '{task.name}' ({task.id}) finalized as FAILED: {task.result} ---")

            # Clean up generator for the completed/failed task
            if task.id in self.task_generators:
                del self.task_generators[task.id]

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
