"""
LLM Agent Scheduler 客户端示例

这个脚本展示了如何使用 LLM Agent Scheduler 服务提交、查询和管理任务。
"""
import asyncio
import json
import random
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import httpx

# 服务基础URL
BASE_URL = "http://localhost:8000"

class TaskClient:
    """任务客户端类，封装与调度器服务的交互"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip('/')
    
    async def submit_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """提交新任务"""
        url = f"{self.base_url}/tasks"
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=task_data, timeout=30.0)
            response.raise_for_status()
            return response.json()
    
    async def get_task(self, task_id: str) -> Dict[str, Any]:
        """获取任务详情"""
        url = f"{self.base_url}/tasks/{task_id}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            return response.json()
    
    async def list_tasks(self, limit: int = 10, **filters) -> List[Dict[str, Any]]:
        """获取任务列表"""
        url = f"{self.base_url}/tasks"
        params = {"limit": limit, **filters}
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            return response.json()
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        url = f"{self.base_url}/stats"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            return response.json()
    
    async def update_strategy(self, strategy: str, time_slice: Optional[float] = None) -> Dict[str, Any]:
        """更新调度策略"""
        url = f"{self.base_url}/scheduler/strategy"
        payload = {"strategy": strategy}
        if time_slice is not None:
            payload["time_slice"] = time_slice
            
        async with httpx.AsyncClient() as client:
            response = await client.put(url, json=payload, timeout=10.0)
            response.raise_for_status()
            return response.json()

# 导入 TaskType 枚举
from src.task import TaskType

# 示例任务数据
SAMPLE_TASKS = [
    # 函数调用任务 - 获取天气
    {
        "name": "获取天气",
        "payload": {
            "function_name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "城市名称，例如：北京、上海"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
                },
                "required": ["location"]
            },
            "content": "今天北京的天气怎么样？",
            "system_prompt": "你是一个天气助手，请根据用户请求获取天气信息。",
            "model": "deepseek-ai/DeepSeek-R1"
        },
        "task_type": TaskType.FUNCTION_CALL.value,  # 使用枚举值
        "priority": 1,
        "estimated_time": 3.0,
        "tags": ["weather"]
    },
    
    # 文本生成任务 - 使用 generate_text 方法
    {
        "name": "生成诗歌",
        "payload": {
            "content": "请写一首关于春天的诗",
            "system_prompt": "你是一个诗人，请用优美的语言创作一首诗。",
            "temperature": 0.8,
            "max_tokens": 200,
            "model": "deepseek-ai/DeepSeek-R1"
        },
        "task_type": "generate_text",  # 自定义任务类型
        "priority": 2,
        "estimated_time": 5.0,
        "tags": ["poetry"]
    },
    
    # 文件操作任务
    {
        "name": "文件处理",
        "payload": {
            "filename": "example.txt",
            "operation": "read",
            "content": "这是文件内容"
        },
        "task_type": TaskType.FILE_OPERATION.value,  # 使用枚举值
        "priority": 0,  # 高优先级
        "estimated_time": 1.5,
        "tags": ["file"]
    }
]

async def demo_submit_tasks():
    """演示如何提交多个任务"""
    client = TaskClient()
    
    print("=== 开始提交任务 ===")
    tasks = []
    for task_data in SAMPLE_TASKS:
        try:
            result = await client.submit_task(task_data)
            task_id = result.get("task_id")
            tasks.append(task_id)
            print(f"✅ 任务已提交 - ID: {task_id}, 名称: {task_data['name']}")
        except Exception as e:
            print(f"❌ 提交任务失败: {str(e)}")
    
    return tasks

async def demo_monitor_tasks(task_ids: List[str]):
    """演示如何监控任务状态"""
    client = TaskClient()
    
    print("\n=== 监控任务状态 ===")
    completed = set()
    task_info_map = {}
    
    while len(completed) < len(task_ids):
        for task_id in task_ids:
            if task_id in completed:
                continue
                
            try:
                task_info = await client.get_task(task_id)
                status = task_info.get("status")
                task_name = task_info.get("name", "未知任务")
                
                # 只在状态变化时显示
                if task_id not in task_info_map or task_info_map[task_id].get("status") != status:
                    if status in ["completed", "failed"]:
                        completed.add(task_id)
                        result = task_info.get("result", {})
                        
                        # 格式化输出结果
                        if status == "completed":
                            print(f"\n✅ 任务完成 - {task_name} (ID: {task_id})")
                            if isinstance(result, dict):
                                if "generated_text" in result:
                                    print(f"   生成内容: {result['generated_text'][:200]}...")
                                elif "status" in result and result["status"] == "success":
                                    print(f"   操作成功: {result}")
                                else:
                                    print(f"   结果: {result}")
                            else:
                                print(f"   结果: {str(result)[:200]}...")
                        else:
                            print(f"\n❌ 任务失败 - {task_name} (ID: {task_id})")
                            print(f"   错误: {task_info.get('error', '未知错误')}")
                    else:
                        print(f"\n⏳ 任务进行中 - {task_name} (ID: {task_id})")
                        print(f"   状态: {status.upper()}")
                        
                task_info_map[task_id] = task_info
                    
            except Exception as e:
                print(f"❌ 获取任务状态失败: {str(e)}")
        
        if len(completed) < len(task_ids):
            await asyncio.sleep(2)  # 每2秒检查一次
    
    print("\n所有任务已完成！")

async def demo_scheduler_stats():
    """演示如何获取调度器统计信息"""
    client = TaskClient()
    
    print("\n=== 调度器统计信息 ===")
    try:
        stats = await client.get_stats()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"获取统计信息失败: {str(e)}")

async def demo_change_scheduler_strategy():
    """演示如何更改调度策略"""
    client = TaskClient()
    
    print("\n=== 更改调度策略 ===")
    try:
        # 更改为时间片轮转策略，时间片为2秒
        result = await client.update_strategy(
            strategy="round_robin",
            time_slice=2.0
        )
        print(f"调度策略已更新: {result}")
    except Exception as e:
        print(f"更新调度策略失败: {str(e)}")

async def main():
    """主函数"""
    print("=" * 50)
    print("LLM Agent Scheduler 客户端示例")
    print("=" * 50)
    
    try:
        # 1. 提交示例任务
        task_ids = await demo_submit_tasks()
        
        # 2. 显示调度器统计信息
        await demo_scheduler_stats()
        
        # 3. 更改调度策略（可选）
        # await demo_change_scheduler_strategy()
        
        # 4. 监控任务状态
        await demo_monitor_tasks(task_ids)
        
        # 5. 显示最终统计信息
        await demo_scheduler_stats()
        
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
    finally:
        print("\n示例程序执行完毕！")

if __name__ == "__main__":
    asyncio.run(main())
