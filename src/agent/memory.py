"""Memory management for the agent.

This module handles conversation history and context management.
"""

from datetime import datetime
from typing import Any


class AgentMemory:
    """Memory system for tracking agent conversations and context."""

    def __init__(self, max_history: int = 100) -> None:
        """Initialize agent memory.
        
        Args:
            max_history: Maximum number of interactions to store.
        """
        self.max_history = max_history
        self.history: list[dict[str, Any]] = []
        self.context: dict[str, Any] = {}

    def add_interaction(
        self,
        user_input: str,
        agent_response: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add an interaction to memory.
        
        Args:
            user_input: User's input text.
            agent_response: Agent's response.
            metadata: Optional metadata about the interaction.
        """
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "agent_response": agent_response,
            "metadata": metadata or {},
        }
        
        self.history.append(interaction)
        
        # Trim history if needed
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_history(self, n: int | None = None) -> list[dict[str, Any]]:
        """Get conversation history.
        
        Args:
            n: Number of recent interactions to return. None returns all.
            
        Returns:
            List of interaction dictionaries.
        """
        if n is None:
            return self.history.copy()
        return self.history[-n:]

    def get_last_interaction(self) -> dict[str, Any] | None:
        """Get the most recent interaction.
        
        Returns:
            Last interaction dictionary or None if empty.
        """
        return self.history[-1] if self.history else None

    def set_context(self, key: str, value: Any) -> None:
        """Set a context variable.
        
        Args:
            key: Context key.
            value: Context value.
        """
        self.context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context variable.
        
        Args:
            key: Context key.
            default: Default value if key not found.
            
        Returns:
            Context value or default.
        """
        return self.context.get(key, default)

    def clear_context(self) -> None:
        """Clear all context variables."""
        self.context.clear()

    def clear(self) -> None:
        """Clear all memory (history and context)."""
        self.history.clear()
        self.context.clear()

    def search_history(self, query: str) -> list[dict[str, Any]]:
        """Search conversation history for matching interactions.
        
        Args:
            query: Search query string.
            
        Returns:
            List of matching interactions.
        """
        query_lower = query.lower()
        matches = []
        
        for interaction in self.history:
            if (query_lower in interaction["user_input"].lower() or
                query_lower in interaction["agent_response"].lower()):
                matches.append(interaction)
        
        return matches

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of memory state.
        
        Returns:
            Dictionary with memory statistics.
        """
        return {
            "total_interactions": len(self.history),
            "max_history": self.max_history,
            "context_keys": list(self.context.keys()),
            "oldest_interaction": self.history[0]["timestamp"] if self.history else None,
            "newest_interaction": self.history[-1]["timestamp"] if self.history else None,
        }
