import os
import time
import autogen
from dotenv import load_dotenv

from common.tools import arxiv_search

# Load environment variables
load_dotenv()

# Ensure API keys are set
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# --- LLM CONFIGURATION ---
config_list = [
    {
        'model': 'gpt-4-turbo',
        'api_key': os.environ['OPENAI_API_KEY'],
    }
]

llm_config = {
    "config_list": config_list,
    "cache_seed": 42, # Use a seed for caching
    "temperature": 0.5,
}

# --- AGENTS DEFINITION ---

# 1. User Proxy Agent (Executes code)
user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False, # Set to True if you have Docker installed
    },
    system_message="A human user who can run code. You will execute function calls suggested by other agents."
)

# 2. Researcher Agent
researcher = autogen.AssistantAgent(
    name="Researcher",
    system_message="You are a research assistant. Your job is to find relevant academic papers. You should suggest calling the `arxiv_search` function. Do not try to write the final report.",
    llm_config=llm_config,
)

# 3. Writer Agent
writer = autogen.AssistantAgent(
    name="Writer",
    system_message="You are a technical writer. Your job is to take the research findings, summarize them, and write a final, cohesive report. You should start your work only after the researcher has provided the search results. End your final report with the word TERMINATE.",
    llm_config=llm_config,
)

# Register the tool with the user proxy agent
user_proxy.register_function(
    function_map={
        "arxiv_search": arxiv_search
    }
)

# --- GROUP CHAT DEFINITION ---
groupchat = autogen.GroupChat(
    agents=[user_proxy, researcher, writer],
    messages=[],
    max_round=15
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# --- EXECUTION ---

if __name__ == "__main__":
    print("--- Starting AutoGen Experiment ---")
    start_time = time.time()

    # The initial prompt that kicks off the conversation
    initial_prompt = (
        "Please research the topic 'Applications of Large Language Models in Software Engineering'. "
        "First, find the 5 most recent and relevant papers on arXiv. "
        "Then, summarize each paper. "
        "Finally, synthesize all summaries into a brief review report."
    )

    user_proxy.initiate_chat(
        manager,
        message=initial_prompt
    )

    end_time = time.time()
    execution_time = end_time - start_time

    final_report = groupchat.messages[-2]['content']

    print("\n--- AutoGen Experiment Finished ---")
    print(f"Total Execution Time: {execution_time:.2f} seconds")
    print("\n--- Final Report ---")
    print(final_report)

    # Log the results to a file
    with open("experiments/results/autogen_result.txt", "w", encoding="utf-8") as f:
        f.write(f"Execution Time: {execution_time:.2f} seconds\n\n")
        f.write(final_report)
