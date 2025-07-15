# main.py
import logging
import os
from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

# Import refactored components
from src.task import Task, TaskType
from src.scheduler import Scheduler
from src.llm_service import LLMService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main")

# --- Component Initialization ---
# Ensure OPENAI_API_KEY is set
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# 1. Create the LLM Service
llm_service = LLMService(api_key=api_key)

# 2. Create the Scheduler, injecting the LLM service
scheduler = Scheduler(llm_service=llm_service, max_concurrent_tasks=5)

# --- FastAPI Application Setup ---
app = FastAPI(
    title="LLM Agent Scheduler",
    description="An OS-inspired asynchronous scheduler for LLM agents with true concurrency.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Be more restrictive in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Application Lifecycle Events ---
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Starting scheduler...")
    await scheduler.start()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown: Stopping scheduler...")
    await scheduler.stop()

# --- Pydantic Models for API ---

class CreateTaskRequest(BaseModel):
    name: str = Field(..., description="A descriptive name for the task.")
    payload: Dict[str, Any] = Field(..., description="The data required for the task, e.g., {'messages': [...], 'tools': [...]}")
    priority: int = 1
    task_type: str = Field(TaskType.FUNCTION_CALL.value, description="The type of the task.")
    is_decomposable: bool = False

class TaskResponse(BaseModel):
    task_id: str
    message: str = "Task successfully submitted and queued."

# --- API Endpoints ---
@app.get("/")
async def root():
    return {
        "message": "LLM Agent Scheduler is running.",
        "version": app.version,
        "docs_url": "/docs"
    }

@app.post("/tasks", response_model=TaskResponse, status_code=202)
async def submit_task(task_request: CreateTaskRequest):
    """Submits a new task to the scheduler."""
    try:
        task_type_enum = TaskType(task_request.task_type.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid task type: '{task_request.task_type}'. Valid types are: {[t.value for t in TaskType]}")

    try:
        task = Task(
            name=task_request.name,
            payload=task_request.payload,
            priority=task_request.priority,
            task_type=task_type_enum,
            is_decomposable=task_request.is_decomposable
        )
        await scheduler.add_task(task)
        logger.info(f"Task submitted: {task.id} - {task.name}")
        return {"task_id": task.id}
    except Exception as e:
        logger.error(f"Failed to submit task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/tasks/{task_id}", response_model=Dict[str, Any])
async def get_task_status(task_id: str = Path(..., description="The ID of the task to retrieve.")):
    """Retrieves the current status and details of a specific task."""
    task = await scheduler.get_task_by_id(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task with ID '{task_id}' not found.")
    return task.to_dict()

@app.get("/stats", response_model=Dict[str, Any])
async def get_scheduler_stats():
    """Gets current statistics from the scheduler."""
    return await scheduler.get_stats()

# --- Main Entry Point ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
