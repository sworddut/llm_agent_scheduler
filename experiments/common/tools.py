import arxiv
import logging
import asyncio
import json
from typing import Union, List, Dict, Any, Callable, Tuple
from fastmcp import Client
from functools import partial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_all_tools() -> Tuple[List[Dict[str, Any]], Dict[str, Callable]]:
    """
    Gathers all available tools from MCP servers and prepares them for the agent.
    """
    with open("src/mcp/mcp_config.json", "r") as f:
        mcp_config = json.load(f)
    
    mcp_client = Client(mcp_config)

    all_definitions = []
    all_callables = {}

    try:
        async with mcp_client:
            # Use the built-in method to get OpenAI-compatible tool definitions
            mcp_tool_defs = await mcp_client.list_tools()
            for tool in mcp_tool_defs:
                # Per user's suggestion, use tool.__dict__ and wrap it for OpenAI format.
                # We also need to rename 'inputSchema' to 'parameters' for OpenAI compatibility.
                tool_as_dict = tool.__dict__
                tool_as_dict['parameters'] = tool_as_dict.pop('inputSchema', {})

                all_definitions.append({
                    "type": "function",
                    "function": tool_as_dict
                })

                # Create a callable for the tool using its name
                tool_name = tool.name
                if tool_name:
                    all_callables[tool_name] = partial(mcp_client.call_tool, name=tool_name)
        logger.info(f"Successfully loaded {len(all_definitions)} tools from MCP.")
    except Exception as e:
        logger.error(f"Failed to load MCP tools: {e}.")

    logger.info(f"Total tools loaded: {len(all_definitions)}. Names: {list(all_callables.keys())}")
    return all_definitions, all_callables

if __name__ == '__main__':
    async def test_tools():
        definitions, callables = await get_all_tools()
        print("--- All Tool Definitions ---")
        print(json.dumps(definitions, indent=2))
        
        print("\n--- Testing a local tool: arxiv_search ---")
        result = callables['arxiv_search'](query="LLM agents")
        print(result[:300] + "...")

        print("\n--- Testing an MCP tool: amap-maps-streamableHTTP.maps_weather ---")
        if 'amap-maps-streamableHTTP.maps_weather' in callables:
            weather_tool = callables['amap-maps-streamableHTTP.maps_weather']
            weather_result = await weather_tool(city="广州")
            print(weather_result)
        else:
            print("MCP weather tool not loaded.")

    asyncio.run(test_tools())
