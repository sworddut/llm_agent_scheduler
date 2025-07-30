# 异步 LLM 智能体调度器 (v1.0 - 静态 DAG 执行器)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Asyncio-green)
![License](https://img.shields.io/badge/License-MIT-blue)

**一个强大的、能够将复杂用户目标自主分解为静态有向无环图 (DAG) 任务，并由高性能异步调度器执行的健壮框架。**

此版本 (`v1.0`) 是一个稳定、强大的基础，用于构建可靠的 AI 智能体，能够处理具有显式依赖管理和并发执行能力的复杂、多步骤工作流。

---

## 1. 🚀 核心哲学：计划与执行 (Plan-and-Execute)

本项目超越了简单的、反应式的智能体循环 (如 ReAct) 的局限性。它实现了一种 **“计划与执行”** 的范式，将“思考”与“行动”分离：

1.  **规划阶段 (Planning)**：一个专门的 `PlannerAgent` 接收一个高级别的用户目标。它利用大语言模型 (LLM) 分析该目标，并将其分解为一个结构化的、机器可读的计划。这个计划就是一个 DAG，其中节点是独立的子任务，边代表依赖关系。

2.  **执行阶段 (Execution)**：`Scheduler` 接收这个静态的 DAG 并执行它。它会尊重所有依赖关系，并并发地运行独立的任务，从而最大化效率和速度。

这种方法为复杂工作流提供了无与伦比的稳定性、可预测性和性能。

---

## 2. 🏛️ 系统架构

该系统构建在一系列解耦的、高内聚的组件之上：

*   **`Task` (`src/task.py`)**: 原子工作单元。每个任务都有一个类型 (`PLANNING`, `TOOL_CALL`, `FINAL_SUMMARY`)、一个状态、依赖关系和有效负载。

*   **`Scheduler` (`src/scheduler.py`)**: 系统的核心。它是一个异步引擎，负责：
    *   管理一个中央任务存储。
    *   解决依赖关系，确保一个任务只有在其前置任务都完成后才运行。
    *   使用 `asyncio.Semaphore` 来控制并发，防止系统过载。
    *   编排任务从 `QUEUED` 到 `COMPLETED` 的整个生命周期。

*   **`PlannerAgent` (`src/agent.py`)**: 高级别的战略思想家。它接收用户提示，并输出一个定义整个任务 DAG 的 JSON 对象。

*   **`Agent` (`src/agent.py`)**: 通用的任务执行者。它通过与 LLM 和外部工具 (通过 MCP) 交互来处理 `TOOL_CALL` 和 `FINAL_SUMMARY` 类型的任务。

*   **`MCP 客户端` (`src/mcp/`)**: 用于模型中心协议 (Model-Centric Protocol) 的客户端，允许智能体以标准化的方式与外部工具 (如网络搜索、API) 交互。

---

## 3. ✨ 此版本主要特性

- **自主任务分解**: `PlannerAgent` 能够从单个用户请求中自主创建复杂的多步骤计划。
- **基于 DAG 的依赖管理**: 完全支持定义和执行具有复杂依赖关系的任务。
- **并发执行**: 当任务的依赖关系得到满足时，异步的 `Scheduler` 会自动并行执行任务，显著加快执行时间。
- **集中式状态管理**: `Scheduler` 作为所有任务状态的唯一真实来源，确保系统的一致性。
- **最终摘要合成**: 系统能够通过综合所有先前工具调用子任务的结果，生成一份连贯的最终报告。

---

## 4. 🛠️ 如何运行

1.  **设置环境**:
    ```bash
    # 创建并激活虚拟环境
    python -m venv .venv
    # 在 Windows 上: .venv\Scripts\activate
    source .venv/bin/activate

    # 安装依赖
    pip install -r requirements.txt
    ```

2.  **配置 API 密钥**:
    *   将 `.env.example` 复制为 `.env`。
    *   填入你的 `OPENAI_API_KEY`。

3.  **运行示例**:
    测试系统的主要入口点是 `experiments/run_our_system.py`。
    ```bash
    python -m experiments.run_our_system
    ```
    该脚本将模拟一个用户请求，并将最终合成的报告打印到控制台。

---

## 5. 🗺️ 后续步骤 (未来分支)

这个稳定版本为实现更高级的功能铺平了道路，这些功能将在新的分支中进行探索：

- **动态与反应式规划**: 演进 `Scheduler` 和 `PlannerAgent`，以支持动态的、有条件的工作流，其中任务图可以根据任务结果在执行过程中被修改或扩展。
- **持久化任务存储**: 集成数据库来存储任务状态，从而支持长时间运行的工作流和故障恢复。
- **增强的错误处理**: 为失败的任务实现健壮的重试机制和回退路径。
