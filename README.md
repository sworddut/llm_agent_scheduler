# Asynchronous LLM Agent Scheduler: A Framework for Autonomous Task Decomposition and Execution

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-green)
![License](https://img.shields.io/badge/License-MIT-blue)

**A next-generation AI agent framework that autonomously decomposes complex tasks into a dependency graph (DAG) and executes them concurrently with a sophisticated asynchronous scheduler.**

---

## 1. 🚀 Project Vision & Motivation

Current mainstream Large Language Model (LLM) Agent frameworks often struggle with complex, multi-step tasks that require intricate dependency management and parallel execution. They typically operate on a linear, reactive loop (e.g., ReAct), which lacks global planning capabilities and leads to inefficiencies and failures when faced with non-linear task flows.

This project introduces a novel **Plan-and-Execute** paradigm. We aim to build an intelligent and robust system that can:

1.  **Autonomously Decompose**: Take a high-level, ambiguous user goal and have an LLM-powered `PlannerAgent` break it down into a structured, machine-readable task graph (a Directed Acyclic Graph, or DAG).
2.  **Manage Complex Dependencies**: Explicitly define and manage dependencies between subtasks, ensuring correct execution order (e.g., Task C can only start after both Task A and B are complete).
3.  **Schedule Concurrently**: Utilize an asynchronous, semaphore-controlled `Scheduler` to execute independent tasks in parallel, maximizing throughput and efficiency.
4.  **Execute Flexibly**: Employ a generic `Agent` that can dynamically adapt its context and toolset to handle various subtask types, from function calls to further reasoning steps.

Our system is designed to be **stable, predictable, and efficient**, moving beyond the limitations of conversational or reactive agents to provide a true workflow automation platform.

---

## 2. 🏛️ System Architecture & Core Components

The system is built around a central, asynchronous scheduling core, interacting with intelligent agents and a task management system.

### Key Components (`src/` directory):

*   **`task.py`**: Defines the fundamental unit of work, the `Task` object.
    *   **Attributes**: `id`, `name`, `status` (e.g., `QUEUED`, `RUNNING`, `COMPLETED`, `WAITING_FOR_SUBTASKS`), `dependencies`, `parent_id`.
    *   **Functionality**: Encapsulates all information required for a task's lifecycle, including its payload, type, and relationships with other tasks.

*   **`scheduler.py`**: The heart of the system.
    *   **Core Logic**: Manages a task queue and uses an `asyncio.Semaphore` to control concurrency.
    *   **`_drive_task` loop**: The main scheduling loop that fetches tasks, checks their dependencies, and dispatches them for execution.
    *   **Dependency Resolution**: Before running a task, it ensures all its dependencies are in the `COMPLETED` state.
    *   **Parent Task Management**: When a parent task is decomposed, it enters a `WAITING_FOR_SUBTASKS` state until all its children are finished.

*   **`agent.py`**: The "brain" and "hands" of the system.
    *   **`PlannerAgent`**: A specialized agent responsible for the initial planning phase. Its `decompose_task` method uses a carefully crafted system prompt to guide an LLM to return a JSON object representing the task DAG.
    *   **`Agent`**: The generic task executor. Its `process_task` method is highly flexible:
        *   **Dynamic Context**: It can start a task from a simple `prompt` or a `tool_name` without requiring a pre-existing `messages` list.
        *   **Dynamic Tool Schema**: For `FUNCTION_CALL` tasks, it dynamically generates the JSON Schema for the tool based on the provided parameters, ensuring the LLM understands how to call the tool correctly.

*   **`main.py`**: The FastAPI server that exposes the system's capabilities via a REST API, allowing clients to submit tasks and monitor their progress.

---

## 3. 💡 How We Differ: A Comparative Analysis

Our architecture provides unique advantages over existing frameworks:

| Framework | Core Paradigm | Our Key Differentiator |
| :--- | :--- | :--- |
| **LangChain Agents** | Reactive Loop (ReAct) | **Proactive Planning**: We generate a global task plan upfront, enabling complex dependency management and parallelism, unlike the linear, step-by-step nature of ReAct. |
| **AutoGen** | Multi-Agent Conversation | **Structured Execution**: We provide a deterministic, task-driven workflow engine, ensuring predictable and stable execution, in contrast to the emergent and often unpredictable nature of conversational flows. |
| **CrewAI** | Role-Based Orchestration | **Autonomous Decomposition**: Our `PlannerAgent` autonomously generates the task plan from a high-level goal, whereas CrewAI typically requires developers to pre-define the tasks and workflow. |


---

## 4. 🔬 Academic & Experimental Plan

To validate the effectiveness and superiority of our approach, we are preparing for a submission to a top-tier AI conference (e.g., AAAI, NeurIPS).

### Core Thesis

An autonomous, DAG-based planning and scheduling system for LLM agents significantly outperforms traditional reactive or conversational models in terms of execution efficiency, stability, and capability to handle complex, non-linear tasks.

### Experimental Setup

We will conduct a comparative study using a representative complex task:

> *"Research the topic 'Applications of Large Language Models in Software Engineering'. First, find the 5 most recent relevant papers on arXiv. Then, summarize each paper. Finally, synthesize all summaries into a brief review report."*

*   **Frameworks for Comparison**: Our System, CrewAI, AutoGen.
*   **Metrics**: End-to-end execution time, total LLM API calls (cost), and implementation complexity.
*   **Location**: All experiment-related code is located in the `/experiments` directory.

---

## 5. 🛠️ Getting Started for Developers

1.  **Environment Setup**:
    ```bash
    # Create and activate a virtual environment
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

    # Install dependencies
    pip install -r requirements.txt
    ```

2.  **Configuration**:
    *   Copy `.env.example` to `.env`.
    *   Fill in your `OPENAI_API_KEY` and, if necessary, the `OPENAI_BASE_URL`.

3.  **Run the Server**:
    ```bash
    uvicorn src.main:app --reload
    ```
    The API documentation will be available at `http://127.0.0.1:8000/docs`.

4.  **Run the Example Client**:
    ```bash
    python example_client.py
    ```

5.  **Run the Experiments**:
    Navigate to the `experiments/` directory to find scripts for running comparative tests.

---

## 6. 🗺️ Future Roadmap

- **[Research]** Complete the comparative experiments and publish the findings.
- **[Feature]** Implement a persistent storage layer (e.g., a database) for tasks to ensure durability.
- **[Feature]** Develop a more robust error handling and retry mechanism for tasks.
- **[Feature]** Build a simple web UI to visualize the task graph and monitor execution progress in real-time.
- **[Feature]** Introduce a memory module for agents to retain context across complex tasks.

## 7. 🤝 Contributing

We welcome contributions! Whether it's improving the core scheduler, adding new agent capabilities, or helping with the experimental analysis, your input is valuable. Please feel free to open an issue or submit a pull request.








