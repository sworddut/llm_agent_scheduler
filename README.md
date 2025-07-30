# Asynchronous LLM Agent Scheduler (v1.0 - Static DAG Executor)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Asyncio-green)
![License](https://img.shields.io/badge/License-MIT-blue)

**A robust framework for autonomously decomposing complex user goals into a static Directed Acyclic Graph (DAG) of tasks and executing them with a high-performance asynchronous scheduler.**

This version (`v1.0`) represents a stable, powerful foundation for building reliable AI agents that can handle intricate, multi-step workflows with explicit dependency management and concurrency.

---

## 1. 🚀 Core Philosophy: Plan-and-Execute

This project moves beyond the limitations of simple, reactive agent loops (like ReAct). It implements a **Plan-and-Execute** paradigm, which separates the "thinking" from the "doing":

1.  **Planning Phase**: A specialized `PlannerAgent` receives a high-level user goal. It leverages a Large Language Model (LLM) to analyze the goal and break it down into a structured, machine-readable plan. This plan is a DAG where nodes are individual subtasks and edges represent dependencies.

2.  **Execution Phase**: The `Scheduler` takes this static DAG and executes it. It respects all dependencies and runs independent tasks concurrently, maximizing efficiency and speed.

This approach provides unparalleled stability, predictability, and performance for complex workflows.

---

## 2. 🏛️ System Architecture

The system is built on a set of decoupled, high-cohesion components:

*   **`Task` (`src/task.py`)**: The atomic unit of work. Each task has a type (`PLANNING`, `TOOL_CALL`, `FINAL_SUMMARY`), a status, dependencies, and a payload.

*   **`Scheduler` (`src/scheduler.py`)**: The core of the system. It's an asynchronous engine that:
    *   Manages a central task store.
    *   Resolves dependencies, ensuring a task only runs when its prerequisites are complete.
    *   Uses an `asyncio.Semaphore` to control concurrency and prevent system overload.
    *   Orchestrates the entire lifecycle of a task from `QUEUED` to `COMPLETED`.

*   **`PlannerAgent` (`src/agent.py`)**: The high-level strategic thinker. It takes a user prompt and outputs a JSON object defining the entire task DAG.

*   **`Agent` (`src/agent.py`)**: The general-purpose task executor. It handles `TOOL_CALL` and `FINAL_SUMMARY` tasks by interacting with LLMs and external tools (via MCP).

*   **`MCP Client` (`src/mcp/`)**: A client for the Model-Centric Protocol, allowing agents to interact with external tools (e.g., web search, APIs) in a standardized way.

---

## 3. ✨ Key Features of This Version

- **Autonomous Task Decomposition**: The `PlannerAgent` can autonomously create complex, multi-step plans from a single user request.
- **DAG-Based Dependency Management**: Full support for defining and executing tasks with intricate dependencies.
- **Concurrent Execution**: The asynchronous `Scheduler` automatically executes tasks in parallel when their dependencies are met, significantly speeding up execution time.
- **Centralized State Management**: The `Scheduler` acts as the single source of truth for the status of all tasks, ensuring consistency.
- **Final Summary Synthesis**: The system can generate a final, coherent report by synthesizing the results from all preceding tool-call subtasks.

---

## 4. 🛠️ How to Run

1.  **Setup Environment**:
    ```bash
    # Create and activate a virtual environment
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

    # Install dependencies
    pip install -r requirements.txt
    ```

2.  **Configure API Keys**:
    *   Copy `.env.example` to `.env`.
    *   Fill in your `OPENAI_API_KEY`.

3.  **Run the Example**:
    The primary entry point for testing the system is `experiments/run_our_system.py`.
    ```bash
    python -m experiments.run_our_system
    ```
    This script will simulate a user request and print the final, synthesized report to the console.

---

## 5. 🗺️ Next Steps (Future Branches)

This stable version paves the way for more advanced capabilities, which will be explored in new branches:

- **Dynamic & Reactive Planning**: Evolving the `Scheduler` and `PlannerAgent` to support dynamic, conditional workflows where the task graph can be modified or extended mid-execution based on task results.
- **Persistent Task Storage**: Integrating a database to store task states, allowing for long-running workflows and recovery.
- **Enhanced Error Handling**: Implementing robust retry mechanisms and fallback paths for failing tasks.
