"""MCP Server for Walkie AI Agent.

This module provides an MCP server that exposes the agent functionality
as tools that can be used by other systems.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .core import WalkieAgent, create_agent_from_config
from .memory import AgentMemory
from .state import AgentStatus


# Global agent instance
_agent: WalkieAgent | None = None


def get_agent() -> WalkieAgent:
    """Get or create the global agent instance."""
    global _agent
    if _agent is None:
        _agent = create_agent_from_config()
    return _agent


# Create MCP server
app = Server("walkie-agent")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available agent tools."""
    return [
        Tool(
            name="execute_task",
            description="Execute a task using the Walkie AI agent. The agent will process the command and return a response.",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command or query to execute",
                    },
                },
                "required": ["command"],
            },
        ),
        Tool(
            name="process_voice_input",
            description="Process voice input through the agent. Transcribes audio and generates a response.",
            inputSchema={
                "type": "object",
                "properties": {
                    "audio_path": {
                        "type": "string",
                        "description": "Path to the audio file to process",
                    },
                },
                "required": ["audio_path"],
            },
        ),
        Tool(
            name="generate_voice_response",
            description="Generate voice audio from text using the agent's TTS system.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to convert to speech",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path where to save the audio file",
                    },
                },
                "required": ["text", "output_path"],
            },
        ),
        Tool(
            name="get_conversation_history",
            description="Get the conversation history from the agent's memory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of recent interactions to return (optional)",
                    },
                },
            },
        ),
        Tool(
            name="get_agent_state",
            description="Get the current state of the agent (status, current task, etc.).",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="clear_memory",
            description="Clear the agent's conversation memory.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="search_history",
            description="Search the conversation history for specific keywords.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="set_context",
            description="Set a context variable in the agent's memory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Context key",
                    },
                    "value": {
                        "type": "string",
                        "description": "Context value (will be JSON parsed if possible)",
                    },
                },
                "required": ["key", "value"],
            },
        ),
        Tool(
            name="get_context",
            description="Get a context variable from the agent's memory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Context key",
                    },
                },
                "required": ["key"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    agent = get_agent()
    
    try:
        if name == "execute_task":
            command = arguments["command"]
            response = await agent.execute_task(command)
            return [TextContent(type="text", text=response)]
        
        elif name == "process_voice_input":
            audio_path = Path(arguments["audio_path"])
            
            if not audio_path.exists():
                return [TextContent(
                    type="text",
                    text=f"Error: Audio file not found: {audio_path}"
                )]
            
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            
            text, response = await agent.process_voice_input(audio_bytes)
            
            result = {
                "transcribed_text": text,
                "agent_response": response,
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "generate_voice_response":
            text = arguments["text"]
            output_path = Path(arguments["output_path"])
            
            audio_bytes = await agent.generate_voice_response(text)
            
            # Save audio to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            
            result = {
                "output_path": str(output_path),
                "audio_size_bytes": len(audio_bytes),
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "get_conversation_history":
            n = arguments.get("n")
            history = agent.get_conversation_history(n=n)
            return [TextContent(type="text", text=json.dumps(history, indent=2))]
        
        elif name == "get_agent_state":
            state = agent.state.to_dict()
            return [TextContent(type="text", text=json.dumps(state, indent=2))]
        
        elif name == "clear_memory":
            agent.clear_memory()
            return [TextContent(type="text", text="Memory cleared successfully")]
        
        elif name == "search_history":
            query = arguments["query"]
            results = agent.memory.search_history(query)
            return [TextContent(type="text", text=json.dumps(results, indent=2))]
        
        elif name == "set_context":
            key = arguments["key"]
            value = arguments["value"]
            
            # Try to parse as JSON
            try:
                parsed_value = json.loads(value)
            except json.JSONDecodeError:
                parsed_value = value
            
            agent.memory.set_context(key, parsed_value)
            return [TextContent(type="text", text=f"Context '{key}' set successfully")]
        
        elif name == "get_context":
            key = arguments["key"]
            value = agent.memory.get_context(key)
            
            if value is None:
                return [TextContent(type="text", text=f"Context '{key}' not found")]
            
            result = {key: value}
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    except Exception as e:
        error_msg = f"Error executing {name}: {str(e)}"
        return [TextContent(type="text", text=error_msg)]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
