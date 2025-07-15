import asyncio
import os
import time
import logging
from dotenv import load_dotenv

from src.scheduler import Scheduler
from src.task import Task, TaskType, TaskStatus
from src.llm_service import LLMService
from experiments.common.tools import arxiv_search_tool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main function to run the experiment for our system."""
    load_dotenv()

    # 1. Setup: LLM Service and Tools
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")

    llm_service = LLMService(api_key=api_key)
    tools = [arxiv_search_tool]

    # 2. Initialize the Scheduler with tools
    scheduler = Scheduler(llm_service=llm_service, tools=tools, max_concurrent_tasks=20)

    # 3. Define the main task
    initial_prompt = (
        "Please research and find 3-5 recent (since 2023) and highly cited papers on the applications of "
        "Large Language Models in Software Engineering. Focus on areas like code generation, testing, and debugging. "
        "Then, synthesize the findings from these papers into a concise report that summarizes the key trends, challenges, and future directions."
    )
    
    main_task = Task(
        name="Main Research Task",
        task_type=TaskType.PLANNING,
        payload={"prompt": initial_prompt},
    )

    # 4. Run the Scheduler and monitor the task
    start_time = time.time()
    await scheduler.start()
    await scheduler.add_task(main_task)

    logger.info(f"Starting main task {main_task.id}...")

    # Monitor the main task until it's done
    while True:
        task_status = scheduler.get_task_status(main_task.id)
        # In our new model, the parent task is 'COMPLETED' only when its subtasks are done.
        # So we can exit the loop once the parent is no longer in a running/waiting state.
        if main_task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            logger.info(f"Main task {main_task.id} finished with status: {main_task.status.value}")
            break
        logger.info(f"Waiting for main task {main_task.id} to complete. Current status: {main_task.status.value}")
        await asyncio.sleep(5)

    end_time = time.time()

    # 5. Report Results
    total_time = end_time - start_time
    final_result = main_task.result

    print("\n---\n")
    print(f"[Our System] Experiment finished in {total_time:.2f} seconds.")
    print("\nFinal Report:\n")
    print(final_result)
    print("\n---")

    # Save the final report to a file
    # Also, ensure the result is a string before writing.
    if isinstance(final_result, dict):
        report_content = json.dumps(final_result, indent=4)
    else:
        report_content = str(final_result) if final_result else "No result generated."

    with open("our_system_final_report.txt", "w", encoding="utf-8") as f:
        f.write(report_content)

    # 6. Shutdown gracefully
    await scheduler.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
