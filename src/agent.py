# src/agent.py
import json
import logging
from typing import Dict, Any, AsyncGenerator, List, Optional

from .task import Task
from .llm_service import LLMService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Agent")

class Agent:
    """
    The Agent is responsible for executing a single task by interacting with the LLM.
    It's designed as a stateful, pausable executor using an async generator.
    """
    def __init__(self, llm_service: LLMService):
        """
        Initializes the Agent.
        :param llm_service: An instance of LLMService to interact with the language model.
        """
        self.llm_service = llm_service

    async def process_task(self, task: Task) -> AsyncGenerator[Dict[str, Any], Any]:
        """
        Processes a task as an asynchronous generator, allowing it to be paused and resumed.
        This generator communicates with the scheduler by yielding tool call requests.
        The scheduler executes the tool and sends the result back into the generator.

        :param task: The task to be processed.
        :yield: A dictionary representing a tool call request.
        :return: The final result of the task when the generator completes.
        """
        payload = task.payload
        model = payload.get("model", self.llm_service.model)
        messages = payload.get("messages")

        # If messages are not provided, construct them from the payload
        if not messages:
            if "prompt" in payload:
                messages = [{"role": "user", "content": payload["prompt"]}]
            elif "tool_name" in payload:
                # This is a direct command to execute a tool. We can simulate a user request.
                messages = [{
                    "role": "user", 
                    "content": f"Execute the following tool call precisely as specified:\n\nTool: `{payload['tool_name']}`\nParameters: {json.dumps(payload.get('parameters', {}), indent=2)}"
                }]
            else:
                logger.error(f"Task {task.id} has an invalid payload: missing 'messages' or 'prompt'. Payload: {payload}")
                task.fail("Invalid payload")
                return

        logger.info(f"Agent starts processing task {task.id} with model {model}. Initial messages: {len(messages)}")

        while True:
            response = None
            try:
                logger.debug(f"Task {task.id}: Sending request to LLM with {len(messages)} messages.")
                # Delegate the call to LLMService, which handles tool definitions automatically
                response = await self.llm_service.chat_completion(
                    model=model,
                    messages=messages,
                )
                response_message = response.choices[0].message
                logger.debug(f"Task {task.id}: Received response from LLM.\nResponse: {response_message}")

            except Exception as e:
                logger.error(f"Task {task.id} failed during LLM API call: {e}", exc_info=True)
                task.fail(f"LLM API call failed: {e}")
                return

            # Append the AI's response to the message history
            messages.append(response_message)

            if response_message.tool_calls:
                logger.info(f"Task {task.id} requests tool call: {response_message.tool_calls[0].function.name}")
                
                # Yield the tool call request to the scheduler in the expected format
                tool_results = yield {"type": "tool_call", "calls": response.choices[0].message.tool_calls}
                
                # Once resumed, append the tool results to the message history
                # The scheduler now returns a list of results, one for each tool call.
                messages.extend(tool_results)
                logger.info(f"Task {task.id} received tool results. Resuming execution.")
            
            else:
                # No tool calls, we have the final answer.
                final_answer = response_message.content
                logger.info(f"Task {task.id} finished. Final answer: {final_answer[:100]}...")
                task.complete(final_answer)
                return


class PlannerAgent:
    """
    The PlannerAgent is responsible for decomposing a complex task into a structured plan
    of subtasks. It interacts with the LLM to generate this plan.
    """
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    async def decompose_task(self, task: "Task") -> dict:
        logger.info(f"PlannerAgent: Decomposing task {task.id} ('{task.name}')")
        user_prompt = task.payload.get("goal")
        system_prompt = await self._get_planning_system_prompt()

        logger.debug(f"--- Planner System Prompt for Task {task.id} ---\n{system_prompt}\n--------------------------------------------------")
        logger.debug(f"--- Planner User Prompt for Task {task.id} ---\n{user_prompt}\n--------------------------------------------------")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # Use the LLMService's chat_completion which handles tools automatically
            response = await self.llm_service.chat_completion(
                model=self.llm_service.model,
                messages=messages,
                response_format={"type": "json_object"}, # Force JSON output
            )
            plan_json_str = response.choices[0].message.content
            logger.debug(f"Planner received LLM response for task {task.id}:\n{plan_json_str}")
            plan = json.loads(plan_json_str)
            logger.debug(f"Parsed plan for task {task.id}: {plan}")
            logger.info(f"PlannerAgent: Successfully decomposed task {task.id}. Plan has {len(plan.get('subtasks', []))} subtasks.")
            # Find the final summary task and add all other tool call tasks as dependencies
            final_summary_task = None
            tool_call_tasks_names = []
            for subtask in plan['subtasks']:
                if subtask['task_type'] == 'FINAL_SUMMARY':
                    final_summary_task = subtask
                elif subtask['task_type'] == 'TOOL_CALL':
                    tool_call_tasks_names.append(subtask['name'])

            if final_summary_task:
                # Ensure dependencies are unique
                existing_deps = set(final_summary_task.get('dependencies', []))
                existing_deps.update(tool_call_tasks_names)
                final_summary_task['dependencies'] = list(existing_deps)
                logger.info(f"Set {len(tool_call_tasks_names)} dependencies for FINAL_SUMMARY task '{final_summary_task['name']}'.")
            return plan
        except json.JSONDecodeError as e:
            logger.error(f"PlannerAgent: Failed to decode JSON plan for task {task.id}. Error: {e}. Response: {plan_json_str}")
            return None
        except Exception as e:
            logger.error(f"PlannerAgent: LLM call failed during planning for task {task.id}. Error: {e}", exc_info=True)
            return None

    async def _get_planning_system_prompt(self) -> str:
        # Fetch tools dynamically from the LLMService
        async with self.llm_service.mcp_client as client:
            tools = await client.list_tools()
        
        tools_for_prompt = []
        for tool in tools:
            tool_dict = {
                'name': tool.name,
                'description': tool.description,
                'parameters': tool.inputSchema
            }
            tools_for_prompt.append(tool_dict)

        tools_json_string = json.dumps(tools_for_prompt, indent=2, ensure_ascii=False)

        prompt = f"""
You are a master planner AI. Your role is to decompose a complex user request into a structured plan of subtasks that can be executed by a machine.
You must respond with a single valid JSON object.

The JSON object must have a single key: "subtasks", which is a list of dictionaries. Each dictionary is a subtask.

Each subtask must have the following structure:
- "name": (string) A unique, descriptive name for the subtask (e.g., "get_weather_for_guangzhou"). This name is used for dependencies.
- "task_type": (string) The type of task. Must be one of:
    - "tool_call": The task executes a tool.
    - "final_summary": The task synthesizes final results. There must be exactly ONE such task, and it must depend on all other tool_call tasks.
- "payload": (object) The data for the task.
    - For "tool_call", it must contain "tool_name" (string) and "parameters" (object).
    - For "final_summary", it must contain a "prompt" (string), which can be an empty string as it will be populated later.
- "dependencies": (list of strings) A list of `name`s of subtasks that must complete before this one starts.

Here are the available tools:
{tools_json_string}

Analyze the user's request and create a logical plan. A final task of type 'final_summary' must be included to provide the final answer to the user, and it should depend on all tool-using tasks.

IMPORTANT: When generating parameters for tool calls, especially for search-related tools, use concise, localized, and native language keywords (e.g., use Chinese for Chinese locations). For example, for a search in Guangzhou, prefer '广州 美食' over 'authentic local food in Guangzhou'. This ensures the best results from the tools.

Example of a valid JSON response:
{{
  "subtasks": [
    {{
      "name": "get_guangzhou_weather",
      "task_type": "tool_call",
      "payload": {{
        "tool_name": "amap-maps-streamableHTTP_maps_weather",
        "parameters": {{
          "city": "Guangzhou"
        }}
      }},
      "dependencies": []
    }},
    {{
      "name": "summarize_and_report",
      "task_type": "final_summary",
      "payload": {{
        "prompt": "Based on the weather information, create a travel suggestion."
      }},
      "dependencies": ["get_guangzhou_weather"]
    }}
  ]
}}

Now, generate the plan for the user's request.
"""
        return prompt
