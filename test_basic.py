"""Basic test without API keys."""

from src.agent.memory import AgentMemory
from src.agent.state import AgentState, AgentStatus


def test_memory():
    """Test memory functionality."""
    print("Testing AgentMemory...")
    memory = AgentMemory(max_history=5)
    
    # Add interactions
    memory.add_interaction("Hello", "Hi there!")
    memory.add_interaction("How are you?", "I'm doing well!")
    
    # Get history
    history = memory.get_history()
    assert len(history) == 2
    print(f"✓ Memory has {len(history)} interactions")
    
    # Search
    results = memory.search_history("hello")
    assert len(results) == 1
    print(f"✓ Search found {len(results)} results")
    
    # Context
    memory.set_context("user", "Alice")
    assert memory.get_context("user") == "Alice"
    print("✓ Context management works")
    
    print("✓ All memory tests passed!\n")


def test_state():
    """Test state functionality."""
    print("Testing AgentState...")
    state = AgentState()
    
    # Initial state
    assert state.status == AgentStatus.IDLE
    print("✓ Initial state is IDLE")
    
    # Set status
    state.set_status(AgentStatus.PROCESSING, "Test task")
    assert state.status == AgentStatus.PROCESSING
    assert state.current_task == "Test task"
    print("✓ Status update works")
    
    # Error handling
    state.set_error("Test error")
    assert state.status == AgentStatus.ERROR
    assert state.last_error == "Test error"
    print("✓ Error handling works")
    
    # Clear error
    state.clear_error()
    assert state.status == AgentStatus.IDLE
    assert state.last_error is None
    print("✓ Error clearing works")
    
    # To dict
    state_dict = state.to_dict()
    assert "status" in state_dict
    assert "uptime_seconds" in state_dict
    print("✓ State serialization works")
    
    print("✓ All state tests passed!\n")


if __name__ == "__main__":
    print("=" * 50)
    print("Running Basic Tests")
    print("=" * 50 + "\n")
    
    test_memory()
    test_state()
    
    print("=" * 50)
    print("✓ All tests completed successfully!")
    print("=" * 50)
