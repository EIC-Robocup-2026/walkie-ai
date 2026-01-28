"""Core agent implementation with LangChain integration.

This module provides the main agent that orchestrates tools like TTS, STT,
navigation, and RAG to handle user commands using tool calling.
"""

from typing import Any, AsyncIterator

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from ..interface.asr import STT
from ..interface.tts import TTS
from .memory import AgentMemory
from .state import AgentState


class WalkieAgent:
    """Main agent for handling voice interactions and tool orchestration."""

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[Tool] | None = None,
        memory: AgentMemory | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the Walkie agent.
        
        Args:
            llm: Language model for agent reasoning.
            tools: List of tools available to the agent.
            memory: Memory system for conversation tracking.
            verbose: Enable verbose logging.
        """
        self.llm = llm
        self.verbose = verbose
        
        # Initialize memory
        self.memory = memory or AgentMemory()
        
        # Initialize state
        self.state = AgentState()
        
        # Initialize tools
        self.tools = tools or self._create_default_tools()
        
        # Bind tools to LLM for tool calling (OpenAI function calling)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Walkie, an AI assistant that helps users with voice interactions.
You have access to various tools to help complete tasks.

Available tools:
{tool_descriptions}

When you need to use a tool:
1. Analyze what the user is asking for
2. Determine which tool(s) can help
3. Call the appropriate tool(s) with correct parameters
4. Use the tool results to formulate your response

Always be helpful and use tools when they can improve your response."""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
        ])

    def _create_default_tools(self) -> list[Tool]:
        """Create default tools for the agent."""
        tools = []
        
        # Text-to-Speech tool
        tts_tool = Tool(
            name="TextToSpeech",
            func=self._text_to_speech_sync,
            description="Convert text to speech audio. Input should be the text to speak.",
        )
        tools.append(tts_tool)
        
        # Speech-to-Text tool
        stt_tool = Tool(
            name="SpeechToText",
            func=self._speech_to_text,
            description="Convert speech audio to text. Input should be audio bytes.",
        )
        tools.append(stt_tool)
        
        return tools

    def _text_to_speech_sync(self, text: str) -> str:
        """Synchronous wrapper for TTS."""
        import asyncio
        
        async def _run():
            tts = TTS()
            audio = await tts.synthesize(text)
            # Store audio in state
            self.state.last_audio_output = audio
            return f"Generated speech audio ({len(audio)} bytes)"
        
        return asyncio.run(_run())

    def _speech_to_text(self, audio_content: bytes) -> str:
        """Convert speech to text."""
        stt = STT()
        text = stt.transcribe(audio_content)
        return text

    async def execute_task(self, command: str) -> str:
        """Execute a task using the agent with tool calling.
        
        Args:
            command: User command or query.
            
        Returns:
            Agent's response.
        """
        # Update state
        self.state.current_task = command
        
        # Get conversation history for context
        history = self.memory.get_history(n=5)
        
        # Build chat history
        chat_history = []
        for interaction in history:
            chat_history.append(HumanMessage(content=interaction["user_input"]))
            chat_history.append(AIMessage(content=interaction["agent_response"]))
        
        # Prepare tool descriptions
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools
        ])
        
        # Invoke LLM with tools
        response = await self.llm_with_tools.ainvoke(
            self.prompt.format_messages(
                input=command,
                chat_history=chat_history,
                tool_descriptions=tool_descriptions,
            )
        )
        
        # Check if tool calls were made
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # Execute tool calls
            tool_results = []
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                
                # Find and execute the tool
                for tool in self.tools:
                    if tool.name == tool_name:
                        if self.verbose:
                            print(f"Calling tool: {tool_name} with args: {tool_args}")
                        result = tool.func(**tool_args) if isinstance(tool_args, dict) else tool.func(tool_args)
                        tool_results.append(f"{tool_name}: {result}")
                        break
            
            # Get final response with tool results
            final_prompt = f"{command}\n\nTool results:\n" + "\n".join(tool_results)
            final_response = await self.llm.ainvoke([
                HumanMessage(content=final_prompt)
            ])
            result = final_response.content
        else:
            # No tool calls, use direct response
            result = response.content
        
        # Store in memory
        self.memory.add_interaction(command, result)
        
        return result

    def execute_task_sync(self, command: str) -> str:
        """Synchronous version of execute_task.
        
        Args:
            command: User command or query.
            
        Returns:
            Agent's response.
        """
        import asyncio
        return asyncio.run(self.execute_task(command))

    async def process_voice_input(self, audio_bytes: bytes) -> tuple[str, str]:
        """Process voice input and generate response.
        
        Args:
            audio_bytes: Audio input as bytes.
            
        Returns:
            Tuple of (transcribed_text, agent_response).
        """
        # Transcribe audio
        stt = STT()
        text = stt.transcribe(audio_bytes)
        
        # Process with agent
        response = await self.execute_task(text)
        
        return text, response

    async def generate_voice_response(self, text: str) -> bytes:
        """Generate voice response from text.
        
        Args:
            text: Text to convert to speech.
            
        Returns:
            Audio bytes.
        """
        tts = TTS()
        audio = await tts.synthesize(text)
        self.state.last_audio_output = audio
        return audio

    async def stream_voice_response(self, text: str) -> AsyncIterator[bytes]:
        """Stream voice response from text.
        
        Args:
            text: Text to convert to speech.
            
        Yields:
            Audio chunks as bytes.
        """
        tts = TTS()
        async for chunk in tts.synthesize_stream(text):
            yield chunk

    def add_tool(self, tool: Tool) -> None:
        """Add a new tool to the agent.
        
        Args:
            tool: LangChain Tool to add.
        """
        self.tools.append(tool)
        # Rebind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def get_conversation_history(self, n: int | None = None) -> list[dict[str, str]]:
        """Get conversation history.
        
        Args:
            n: Number of recent interactions to return. None returns all.
        
        Returns:
            List of conversation turns.
        """
        return self.memory.get_history(n=n)

    def clear_memory(self) -> None:
        """Clear conversation memory."""
        self.memory.clear()


def create_agent_from_config(
    config_path: str | None = None,
    verbose: bool = False,
) -> WalkieAgent:
    """Create agent from configuration file.
    
    Args:
        config_path: Path to config file (defaults to .env).
        verbose: Enable verbose logging.
    
    Returns:
        Configured WalkieAgent instance.
    """
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    
    # Load environment variables
    if config_path:
        load_dotenv(config_path)
    else:
        # Try to find .env in project root
        current = Path(__file__).resolve()
        for parent in current.parents:
            env_path = parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)
                break
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    # Create agent
    agent = WalkieAgent(
        llm=llm,
        verbose=verbose or os.getenv("AGENT_VERBOSE", "false").lower() == "true",
    )
    
    return agent
