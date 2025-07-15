import os
import time
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from common.tools import arxiv_search

# Load environment variables
load_dotenv()

# Ensure API keys are set
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# --- AGENTS DEFINITION ---

# 1. Researcher Agent
researcher = Agent(
    role='Senior Research Analyst',
    goal='Find the most relevant and recent academic papers on a given topic.',
    backstory=(
        "You are a meticulous research analyst with a knack for uncovering "
        "the most pivotal academic papers. You are an expert in using the arXiv library to find scientific articles."
    ),
    tools=[arxiv_search],
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(model_name="gpt-4-turbo", temperature=0.2)
)

# 2. Writer Agent
writer = Agent(
    role='Professional Academic Writer',
    goal='Summarize research findings and compose a clear, concise final report.',
    backstory=(
        "You are a renowned academic writer, known for your ability to distill complex topics "
        "into easy-to-understand summaries and reports. You transform raw research data into polished, insightful articles."
    ),
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(model_name="gpt-4-turbo", temperature=0.7)
)

# --- TASKS DEFINITION ---

# Task 1: Search for papers
search_task = Task(
    description=(
        "Search for the 5 most recent and relevant academic papers on the topic: "
        "'Applications of Large Language Models in Software Engineering'."
    ),
    expected_output=(
        "A formatted string containing the search results for 5 papers, "
        "each including the title, authors, summary, and URL."
    ),
    agent=researcher
)

# Task 2: Summarize the papers
summarize_task = Task(
    description=(
        "For each of the 5 papers found, read its summary and write a one-paragraph summary highlighting its core contribution. "
        "The final output should be a list of these 5 summaries."
    ),
    expected_output=(
        "A list of 5 paragraphs, where each paragraph is a concise summary of one paper."
    ),
    agent=writer,
    context=[search_task] # This task depends on the output of the search_task
)

# Task 3: Write the final report
report_task = Task(
    description=(
        "Combine the 5 summaries into a single, cohesive review report. "
        "The report should have a brief introduction, followed by the summaries, and a concluding sentence."
    ),
    expected_output=(
        "A single, well-formatted report document that synthesizes all the paper summaries."
    ),
    agent=writer,
    context=[summarize_task] # This task depends on the summaries
)

# --- CREW DEFINITION ---

research_crew = Crew(
    agents=[researcher, writer],
    tasks=[search_task, summarize_task, report_task],
    process=Process.sequential, # Tasks will be executed one after another
    verbose=2
)

# --- EXECUTION ---

if __name__ == "__main__":
    print("--- Starting CrewAI Experiment ---")
    start_time = time.time()

    # Kick off the crew's work
    result = research_crew.kickoff()

    end_time = time.time()
    execution_time = end_time - start_time

    print("\n--- CrewAI Experiment Finished ---")
    print(f"Total Execution Time: {execution_time:.2f} seconds")
    print("\n--- Final Report ---")
    print(result)

    # You can also log the results to a file
    with open("experiments/results/crewai_result.txt", "w", encoding="utf-8") as f:
        f.write(f"Execution Time: {execution_time:.2f} seconds\n\n")
        f.write(result)
