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
        tools = payload.get("tools", [])
        model = payload.get("model", self.llm_service.model)
        messages = payload.get("messages", [])

        # If messages are not provided, construct them from the payload
        if not messages:
            if "prompt" in payload:
                messages = [{"role": "user", "content": payload["prompt"]}]
            elif "tool_name" in payload:
                # This is a direct command to execute a tool. We can simulate a user request.
                messages = [{
                    "role": "user", 
                    "content": f"Please call the tool `{payload['tool_name']}` with the following parameters: {json.dumps(payload.get('parameters', {}))}"
                }]
                if not tools:
                    # We need to construct the tool definition for the LLM
                    tools = [{
                        "type": "function",
                        "function": {
                            "name": payload['tool_name'],
                            "description": payload.get('description', 'A tool to be executed.'),
                            "parameters": {
                                "type": "object",
                                "properties": { 
                                    param: {"type": "string", "description": f"Parameter for {payload['tool_name']}"} 
                                    for param in payload.get('parameters', {}).keys()
                                },
                                "required": list(payload.get('parameters', {}).keys()),
                            },
                        }
                    }]
            else:
                logger.error(f"Task {task.id} has an invalid payload: missing 'messages' or 'prompt'. Payload: {payload}")
                task.fail("Invalid payload")
                return

        logger.info(f"Agent starts processing task {task.id} with model {model}. Initial messages: {len(messages)}")

        while True:
            response_message = None
            try:
                logger.debug(f"Task {task.id}: Sending request to LLM with {len(messages)} messages.")
                response = await self.llm_service.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools if tools else None,
                    tool_choice="auto" if tools else None,
                )
                response_message = response.choices[0].message

            except Exception as e:
                logger.error(f"Task {task.id} failed during LLM API call: {e}", exc_info=True)
                task.fail(f"LLM API call failed: {e}")
                return

            # Append the AI's response to the message history
            messages.append(response_message)

            if response_message.tool_calls:
                logger.info(f"Task {task.id} requests tool call: {response_message.tool_calls[0].function.name}")
                
                # Yield the tool call request to the scheduler
                tool_results = yield {"type": "tool_call", "calls": response_message.tool_calls}
                
                # Once resumed, append the tool results to the message history
                for i, tool_call in enumerate(response_message.tool_calls):
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": str(tool_results[i]),  # Ensure content is a string
                        }
                    )
                logger.info(f"Task {task.id} received tool results. Resuming execution.")
            
            else:
                # No tool calls, we have the final answer.
                final_answer = response_message.content
                logger.info(f"Task {task.id} finished. Final answer: {final_answer[:150]}...")
                task.complete(final_answer)
                return


class PlannerAgent:
    """
    The PlannerAgent is responsible for decomposing a complex task into a structured plan
    of subtasks. It interacts with the LLM to generate this plan.
    """
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    async def decompose_task(self, task: Task) -> Optional[Dict[str, Any]]:
        """
        Decomposes a complex user request into a structured plan of subtasks.

        :param task: The high-level task to decompose.
        :return: A dictionary representing the execution plan, or None if planning fails.
        """
        user_prompt = task.payload.get("prompt")
        if not user_prompt:
            logger.error(f"PlannerAgent: Task {task.id} is missing 'prompt' in payload.")
            return None

        system_prompt = self._get_planning_system_prompt()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            logger.info(f"PlannerAgent: Decomposing task {task.id} ('{task.name}')")
            response = await self.llm_service.client.chat.completions.create(
                model=self.llm_service.model,
                messages=messages,
                response_format={"type": "json_object"}, # Force JSON output
            )
            plan_json_str = response.choices[0].message.content
            plan = json.loads(plan_json_str)
            logger.info(f"PlannerAgent: Successfully decomposed task {task.id}. Plan has {len(plan.get('subtasks', []))} subtasks.")
            return plan
        except json.JSONDecodeError as e:
            logger.error(f"PlannerAgent: Failed to decode JSON plan for task {task.id}. Error: {e}. Response: {plan_json_str}")
            return None
        except Exception as e:
            logger.error(f"PlannerAgent: LLM call failed during planning for task {task.id}. Error: {e}", exc_info=True)
            return None

    def _get_planning_system_prompt(self) -> str:
        return """
        You are a master planner AI. Your role is to decompose a complex user request into a series of manageable subtasks.
        You must return the plan as a valid JSON object.

        The JSON object should have a single key: "subtasks".
        The value of "subtasks" should be an array of subtask objects.

        Each subtask object must have the following fields:
        - "name": A short, descriptive name for the subtask (e.g., "search_for_llm_history").
        - "task_type": The type of the subtask. Must be one of: ["REASONING", "FUNCTION_CALL"].
        - "payload": A dictionary containing the necessary data for the subtask. For a REASONING task, this should be a dictionary like `{"prompt": "Detailed question for the subtask..."}`. For a FUNCTION_CALL task, it should contain the arguments.
        - "dependencies": An array of names of other subtasks that must be completed before this one can start. If there are no dependencies, provide an empty array [].

        Analyze the user's request carefully and create a logical, efficient plan. Ensure the dependency graph is correct.

        Here is an example of the expected output format:
        ```json
        {
          "subtasks": [
            {
              "name": "find_hotel",
              "task_type": "REASONING",
              "payload": {
                "prompt": "Search for hotels in Tokyo's Shinjuku or Shibuya districts that are highly rated and have good access to public transport."
              },
              "dependencies": []
            },
            {
              "name": "create_itinerary",
              "task_type": "REASONING",
              "payload": {
                "prompt": "Create a 7-day itinerary with 2-3 daily activities, ensuring a mix of cultural sites, shopping, and food experiences."
              },
              "dependencies": ["find_hotel"]
            }
          ]
        }
        ```

        Example Request: "Write a blog post about the history of the Eiffel Tower and its current visitor statistics."

        Example JSON Output:
        {
            "subtasks": [
                {
                    "name": "research_history",
                    "task_type": "FUNCTION_CALL",
                    "payload": {
                        "tool_name": "web_search",
                        "parameters": {"query": "history of the Eiffel Tower"}
                    },
                    "dependencies": []
                },
                {
                    "name": "research_visitor_stats",
                    "task_type": "FUNCTION_CALL",
                    "payload": {
                        "tool_name": "web_search",
                        "parameters": {"query": "Eiffel Tower visitor statistics 2023"}
                    },
                    "dependencies": []
                },
                {
                    "name": "draft_post",
                    "task_type": "REASONING",
                    "payload": {
                        "prompt": "Using the provided context on its history and visitor statistics, write a compelling blog post about the Eiffel Tower."
                    },
                    "dependencies": ["research_history", "research_visitor_stats"]
                }
            ]
        }
        """
