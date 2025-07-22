import asyncio
from fastmcp import Client
import json

# read JSON config
with open("src/mcp/mcp_config.json", "r") as f:
    config = json.load(f)


client = Client(config)

# Local Python script
# client = Client("src/mcp/mcp_server.py")
async def main():
    async with client:
        # Basic server interaction
        await client.ping()
        
        # List available operations
        tools = await client.list_tools()
        
        # Execute operations
        result = await client.call_tool("maps_weather", {"city": "广州"})
        print(result)

asyncio.run(main())