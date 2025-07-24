import asyncio
from src.llm_service import LLMService


# Local Python script
# client = Client("src/mcp/mcp_server.py")
async def main():
    llm_service = LLMService()

    # 2. Formulate a prompt for the LLM
    user_query = "查询广州今天的天气"
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
    # 4. Process the response and execute the tool using the encapsulated method
    print("\n--- Processing response and executing tool... ---")
    result = await llm_service.process_and_call_tool(
        llm_response=llm_response,
    )

    print("\n--- Final Result ---")
    print(result)

asyncio.run(main())