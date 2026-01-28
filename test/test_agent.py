"""Tests for the Walkie agent."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.agent.core import WalkieAgent
from src.agent.memory import AgentMemory
from src.agent.state import AgentState, AgentStatus


class TestAgentMemory:
    """Test cases for AgentMemory."""

    def test_add_interaction(self):
        """Test adding interactions to memory."""
        memory = AgentMemory(max_history=5)
        
        memory.add_interaction("Hello", "Hi there!")
        assert len(memory.history) == 1
        
        interaction = memory.get_last_interaction()
        assert interaction["user_input"] == "Hello"
        assert interaction["agent_response"] == "Hi there!"

    def test_max_history_limit(self):
        """Test that history is limited to max_history."""
        memory = AgentMemory(max_history=3)
        
        for i in range(5):
            memory.add_interaction(f"Input {i}", f"Response {i}")
        
        assert len(memory.history) == 3
        # Should keep the last 3
        assert memory.history[0]["user_input"] == "Input 2"

    def test_search_history(self):
        """Test searching conversation history."""
        memory = AgentMemory()
        
        memory.add_interaction("What's the weather?", "It's sunny")
        memory.add_interaction("Tell me a joke", "Why did the chicken...")
        memory.add_interaction("Weather forecast", "Tomorrow will be rainy")
        
        results = memory.search_history("weather")
        assert len(results) == 2

    def test_context_management(self):
        """Test context variable management."""
        memory = AgentMemory()
        
        memory.set_context("user_name", "Alice")
        memory.set_context("location", "New York")
        
        assert memory.get_context("user_name") == "Alice"
        assert memory.get_context("location") == "New York"
        assert memory.get_context("missing", "default") == "default"
        
        memory.clear_context()
        assert memory.get_context("user_name") is None


class TestAgentState:
    """Test cases for AgentState."""

    def test_initial_state(self):
        """Test initial agent state."""
        state = AgentState()
        
        assert state.status == AgentStatus.IDLE
        assert state.current_task is None
        assert state.last_error is None

    def test_set_status(self):
        """Test setting agent status."""
        state = AgentState()
        
        state.set_status(AgentStatus.PROCESSING, "Test task")
        assert state.status == AgentStatus.PROCESSING
        assert state.current_task == "Test task"

    def test_error_handling(self):
        """Test error state management."""
        state = AgentState()
        
        state.set_error("Test error")
        assert state.status == AgentStatus.ERROR
        assert state.last_error == "Test error"
        
        state.clear_error()
        assert state.status == AgentStatus.IDLE
        assert state.last_error is None

    def test_is_busy(self):
        """Test busy state detection."""
        state = AgentState()
        
        assert not state.is_busy()
        
        state.set_status(AgentStatus.PROCESSING)
        assert state.is_busy()
        
        state.set_status(AgentStatus.IDLE)
        assert not state.is_busy()

    def test_to_dict(self):
        """Test state serialization."""
        state = AgentState()
        state.set_status(AgentStatus.PROCESSING, "Test task")
        
        state_dict = state.to_dict()
        assert state_dict["status"] == "processing"
        assert state_dict["current_task"] == "Test task"
        assert "uptime_seconds" in state_dict


class TestWalkieAgent:
    """Test cases for WalkieAgent."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = Mock()
        llm.invoke = AsyncMock(return_value="Test response")
        return llm

    def test_agent_initialization(self, mock_llm):
        """Test agent initialization."""
        agent = WalkieAgent(llm=mock_llm)
        
        assert agent.llm == mock_llm
        assert isinstance(agent.memory, AgentMemory)
        assert isinstance(agent.state, AgentState)
        assert len(agent.tools) > 0

    def test_add_tool(self, mock_llm):
        """Test adding a tool to the agent."""
        from langchain_core.tools import Tool
        
        agent = WalkieAgent(llm=mock_llm)
        initial_tool_count = len(agent.tools)
        
        new_tool = Tool(
            name="TestTool",
            func=lambda x: "test",
            description="A test tool",
        )
        
        agent.add_tool(new_tool)
        assert len(agent.tools) == initial_tool_count + 1

    def test_get_conversation_history(self, mock_llm):
        """Test retrieving conversation history."""
        agent = WalkieAgent(llm=mock_llm)
        
        agent.memory.add_interaction("Hello", "Hi!")
        history = agent.get_conversation_history()
        
        assert len(history) == 1
        assert history[0]["user_input"] == "Hello"

    def test_clear_memory(self, mock_llm):
        """Test clearing agent memory."""
        agent = WalkieAgent(llm=mock_llm)
        
        agent.memory.add_interaction("Hello", "Hi!")
        assert len(agent.memory.history) == 1
        
        agent.clear_memory()
        assert len(agent.memory.history) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
