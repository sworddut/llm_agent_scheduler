# LLM Agent Scheduler v0.2.0

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-green)
![OpenAI](https://img.shields.io/badge/OpenAI-API-orange)
![License](https://img.shields.io/badge/License-MIT-blue)

🚀 一个受操作系统调度启发的 LLM Agent 异步任务调度系统，支持多级优先队列、Function Call 任务管理和事件驱动处理。适用于构建更高效的多任务 LLM 系统。

---

## 🌟 项目亮点

- 🧠 **Function Call = Task**：每个 LLM 的 Function Call 被视为一个调度任务
- 🎯 **多级调度策略**：支持优先级调度、时间片轮转、抢占式和最短作业优先等多种策略
- ⚡ **并发处理**：支持多任务并行执行，提高系统吞吐量
- 🔌 **完整 REST API**：基于 FastAPI 构建的全功能 API，支持任务管理和监控
- 🤖 **OpenAI 集成**：直接集成 OpenAI API 实现真实的 Function Call 处理
- 📊 **任务统计**：提供详细的任务执行统计和性能指标
- 🧩 **可扩展架构**：支持自定义任务类型和处理器

---

## 📁 项目结构

```
llm_agent_scheduler/
├── .venv/                # 虚拟环境目录
├── .env                  # 存储 OpenAI API Key 等环境变量
├── main.py               # 主服务入口
├── requirements.txt      # 安装依赖列表
├── src/
│   ├── agent.py          # Agent 实现，处理任务执行和 OpenAI 集成
│   ├── scheduler.py      # 调度器实现，支持多种调度策略
│   └── task.py           # 任务定义，包含任务状态和生命周期管理
```

---

## ✨ 主要功能

### 多种调度策略

- **优先级调度**：基于任务优先级的调度
- **时间片轮转**：为每个任务分配时间片，支持公平调度
- **抢占式调度**：高优先级任务可以抢占低优先级任务
- **最短作业优先**：优先执行预估执行时间最短的任务

### 任务类型

- **Function Call**：执行 OpenAI Function Call 调用
- **API 请求**：执行外部 API 调用
- **文件操作**：处理文件读写操作
- **自定义任务**：支持扩展自定义任务类型

### API 接口

- **提交任务**：`POST /tasks`
- **查询任务列表**：`GET /tasks`
- **查询任务详情**：`GET /tasks/{task_id}`
- **获取统计信息**：`GET /stats`
- **更新调度策略**：`PUT /scheduler/strategy`

---

## ✅ 快速开始

### 1. 安装 [uv](https://github.com/astral-sh/uv)

```bash
# Windows Powershell 安装
irm https://astral.sh/uv/install.ps1 | iex
```

安装后请将 uv 所在路径添加到系统环境变量中。

### 2. 创建并激活虚拟环境

```bash
uv venv
.venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
uv pip install -r requirements.txt
```

或者安装核心依赖：

```bash
uv pip install fastapi uvicorn openai python-dotenv
uv pip freeze > requirements.txt
```

### 4. 添加环境变量 .env

```ini
OPENAI_API_KEY=sk-xxx-your-key
```

### 5. 启动服务

```bash
uvicorn main:app --reload
```

服务启动后，访问 http://localhost:8000/docs 查看 API 文档。

---

## 📝 使用示例

### 提交任务

```bash
curl -X 'POST' \
  'http://localhost:8000/tasks' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "name": "天气查询",
  "payload": {
    "function_name": "get_weather",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
      },
      "required": ["location"]
    },
    "content": "请查询北京的天气",
    "model": "gpt-3.5-turbo"
  },
  "priority": 0,
  "task_type": "function_call",
  "estimated_time": 1.5
}'
```

### 查询任务列表

```bash
curl -X 'GET' 'http://localhost:8000/tasks?limit=5'
```

### 更改调度策略

```bash
curl -X 'PUT' \
  'http://localhost:8000/scheduler/strategy' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "strategy": "round_robin",
  "time_slice": 1.0
}'
```

---

## 🔧 下一步计划

- 📊 **Web UI 仪表盘**：可视化任务流和调度状态
- 💾 **持久化存储**：添加数据库支持，持久化任务和状态
- 🔄 **任务编排**：支持任务依赖和工作流
- 🔐 **认证与授权**：添加 API 访问控制
- 📈 **性能基准测试**：评估不同调度策略的性能
- 🌐 **分布式执行**：支持跨多节点的任务分发

## 📚 学术参考

本项目灵感来自：

- 操作系统任务调度（Multilevel Feedback Queue）
- LangChain, AutoGen 等 Agent 框架设计
- ChatGPT Function Call 机制与消息流控制

## 🧠 你可以做什么？

- ✅ 实现 Web UI 可视化任务调度流
- ✅ 扩展更多任务类型和处理器
- ✅ 对比不同调度策略对 LLM 响应质量的影响
- ✅ 添加更多的单元测试和集成测试

## 📫 贡献

欢迎提交 Issue 和 Pull Request 一同探索 LLM Agent 的更优调度方式！

## 📄 许可证

[MIT License](LICENSE)








