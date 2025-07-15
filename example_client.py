"""
Client example for the refactored LLM Agent Scheduler.

This script demonstrates how to interact with the new OS-like scheduler,
submit tasks that require tool calls, and monitor their lifecycle to observe
the pause/resume mechanism.
"""
import asyncio
import httpx
import logging
import time
from typing import Dict, Any, List

# --- Configuration ---
BASE_URL = "http://localhost:8000"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API Client ---
class TaskClient:
    """A simple client to interact with the scheduler's API."""
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip('/')

    async def submit_task(self, name: str, payload: dict, priority: int = 1, task_type: str = "FUNCTION_CALL", is_decomposable: bool = False) -> Dict[str, Any]:
        """Submits a new task via POST /tasks."""
        url = f"{self.base_url}/tasks"
        task_data = {
            "name": name,
            "payload": payload,
            "priority": priority,
            "task_type": task_type,
            "is_decomposable": is_decomposable
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=task_data, timeout=30.0)
            response.raise_for_status()
            return response.json()

    async def get_task(self, task_id: str) -> Dict[str, Any]:
        """Retrieves task details via GET /tasks/{task_id}."""
        url = f"{self.base_url}/tasks/{task_id}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            return response.json()

    async def get_stats(self) -> Dict[str, Any]:
        """Gets scheduler stats via GET /stats."""
        url = f"{self.base_url}/stats"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            return response.json()

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Gets task status via GET /tasks/{task_id}."""
        url = f"{self.base_url}/tasks/{task_id}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            return response.json()

    async def get_scheduler_stats(self) -> Dict[str, Any]:
        """Gets scheduler stats via GET /stats."""
        url = f"{self.base_url}/stats"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            return response.json()

# --- Test Task Definitions ---

# This task is designed to trigger the scheduler's pause/resume functionality.
# 1. It asks a question that requires a tool.
# 2. The Agent will yield a 'tool_call' request.
# 3. The Scheduler will pause the task, execute the (mock) tool.
# 4. The Scheduler will resume the task with the tool's result.
# 5. The Agent will use the result to formulate the final answer.
TASK_WITH_TOOL_CALL = {
    "name": "Weather and Poem Task",
    "payload": {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that first checks the weather and then writes a short, creative poem about it."},
            {"role": "user", "content": "What's the weather like in Boston today?"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
    }
}

# A simple task without any tools for comparison.
TASK_WITHOUT_TOOL_CALL = {
    "name": "Simple Greeting Task",
    "payload": {
        "messages": [
            {"role": "user", "content": "Write a short, friendly greeting in the style of a pirate."}
        ]
    }
}

# --- Demo Functions ---

async def monitor_task_lifecycle(client: TaskClient, task_id: str):
    """Monitors a single task until it completes, showing its status changes."""
    last_status = None
    while True:
        try:
            task = await client.get_task(task_id)
            status = task.get('status')

            if status != last_status:
                logging.info(f"Task '{task['name']}' ({task_id[:8]}...): Status changed to -> {status}")
                last_status = status

            if status in ["completed", "failed"]:
                logging.info(f"Task '{task['name']}' finished with status {status}.")
                logging.info(f"  - Final Result: {task.get('result')}")
                if task.get('error'):
                    logging.error(f"  - Error: {task.get('error')}")
                break

        except httpx.HTTPStatusError as e:
            logging.error(f"Error fetching task {task_id}: {e}")
            break
        
        await asyncio.sleep(1)

async def main():
    """Main function to run the client demonstration for decomposable tasks."""
    client = TaskClient()
    logging.info("--- LLM Agent Scheduler Client --- Starting Demo ---")

    try:
        await client.get_stats()
        logging.info("Scheduler is online.")
    except httpx.ConnectError:
        logging.error("Connection failed. Is the server running at http://localhost:8000?")
        return

    # Define the user's high-level request for the planner.
    # This will be sent as the 'prompt' inside the payload.
    decomposable_task_prompt = (
        "Create a detailed 7-day travel plan for Tokyo. The plan should include: "
        "1. A recommendation for a hotel in a convenient area like Shinjuku or Shibuya. "
        "2. A daily itinerary with 2-3 activities per day. "
        "3. Suggestions for transportation between locations. "
        "4. A list of 5 must-try Japanese dishes."
    )

    # The payload for a PLANNING task should be a simple dictionary
    # with a 'prompt' key, matching what the PlannerAgent expects.
    payload = {"prompt": decomposable_task_prompt}

    logging.info("--- Submitting Decomposable Task: Plan Tokyo Trip ---")
    response = await client.submit_task(
        name="Plan Tokyo Trip",
        payload=payload,
        task_type="PLANNING"
    )
    task_id = response.get('task_id')

    if not task_id:
        logging.error("Failed to submit the decomposable task.")
        return

    logging.info(f"Submitted parent task with ID: {task_id}")
    logging.info("--- Monitoring Parent Task Lifecycle --- (Will wait for all subtasks to complete)")
    
    await monitor_task_lifecycle(client, task_id)

    logging.info("--- Final Scheduler Stats ---")
    final_stats = await client.get_stats()
    logging.info(final_stats)
    logging.info("--- Demo Finished ---")

if __name__ == "__main__":
    asyncio.run(main())
