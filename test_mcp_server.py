"""Test script for the MCP server.

This script demonstrates how to interact with the Walkie AI agent MCP server.
"""

import asyncio
import json
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_mcp_server():
    """Test the MCP server functionality."""
    
    # Configure server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "src.agent.mcp_server"],
        env={"PYTHONPATH": str(Path(__file__).parent)},
    )
    
    print("Starting MCP server...")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize session
            await session.initialize()
            print("✓ MCP server initialized")
            
            # List available tools
            tools = await session.list_tools()
            print(f"\n✓ Available tools: {len(tools.tools)}")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")
            
            # Test 1: Get agent state
            print("\n--- Test 1: Get Agent State ---")
            result = await session.call_tool("get_agent_state", {})
            state = json.loads(result.content[0].text)
            print(f"Agent status: {state['status']}")
            print(f"Uptime: {state['uptime_seconds']:.2f}s")
            
            # Test 2: Set context
            print("\n--- Test 2: Set Context ---")
            result = await session.call_tool(
                "set_context",
                {"key": "test_user", "value": "Alice"}
            )
            print(result.content[0].text)
            
            # Test 3: Get context
            print("\n--- Test 3: Get Context ---")
            result = await session.call_tool(
                "get_context",
                {"key": "test_user"}
            )
            context = json.loads(result.content[0].text)
            print(f"Context value: {context}")
            
            # Test 4: Execute task (requires OpenAI API key)
            print("\n--- Test 4: Execute Task ---")
            try:
                result = await session.call_tool(
                    "execute_task",
                    {"command": "Say hello in a friendly way"}
                )
                print(f"Agent response: {result.content[0].text}")
            except Exception as e:
                print(f"Note: Task execution requires valid API keys: {e}")
            
            # Test 5: Get conversation history
            print("\n--- Test 5: Get Conversation History ---")
            result = await session.call_tool(
                "get_conversation_history",
                {"n": 5}
            )
            history = json.loads(result.content[0].text)
            print(f"History entries: {len(history)}")
            
            # Test 6: Clear memory
            print("\n--- Test 6: Clear Memory ---")
            result = await session.call_tool("clear_memory", {})
            print(result.content[0].text)
            
            print("\n✓ All tests completed successfully!")


async def main():
    """Main entry point."""
    try:
        await test_mcp_server()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
