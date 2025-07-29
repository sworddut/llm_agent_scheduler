import asyncio
from src.llm_service import LLMService


# Local Python script
# client = Client("src/mcp/mcp_server.py")
async def main():
    llm_service = LLMService()

    # 2. Formulate a prompt for the LLM
    user_query = "查询广州的美食"
    messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. You have access to a set of tools. Based on the user's query, decide which tool to call. Respond with a single JSON object containing 'name' and 'arguments'."
            },
            {
                "role": "user",
                "content": user_query
            }
        ]

    print(f"--- User Query: {user_query} ---")

    # 3. Call the LLM to get the tool call decision
    print("\n--- Asking LLM which tool to use... ---")
    llm_response = await llm_service.chat_completion(
        messages=messages,
        model="gemini-2.5-flash"
    )
    print('llm_response', llm_response)
            # 获取可用工具列表并格式化为OpenAI API兼容的格式
    async with llm_service.mcp_client:
        mcp_tools = await llm_service.mcp_client.list_tools()
    
    # The 'tools' parameter expects a list of dicts, not a JSON string.
    # We also need to rename 'inputSchema' to 'parameters' for OpenAI compatibility.
    tools_for_openai = []
    for tool in mcp_tools:
        tool_as_dict = tool.__dict__
        tool_as_dict['parameters'] = tool_as_dict.pop('inputSchema', {})
        tools_for_openai.append({
            "type": "function",
            "function": tool_as_dict
        })
    # 4. Process the response and execute the tool using the encapsulated method
    print("\n--- Processing response and executing tool... ---")
    tool_call = llm_response.choices[0].message.tool_calls[0]
    result = await llm_service.process_and_call_tool(
        tool_call=tool_call
    )

    print("\n--- Final Result ---")
    print(result)

asyncio.run(main())