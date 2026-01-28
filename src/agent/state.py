"""State management for the agent.

This module tracks the current state of the agent during execution.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass
class AgentState:
    """State container for the agent."""
    
    status: AgentStatus = AgentStatus.IDLE
    current_task: str | None = None
    last_audio_output: bytes | None = None
    last_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    
    def set_status(self, status: AgentStatus, task: str | None = None) -> None:
        """Update agent status.
        
        Args:
            status: New status.
            task: Optional task description.
        """
        self.status = status
        if task:
            self.current_task = task

    def set_error(self, error: str) -> None:
        """Set error state.
        
        Args:
            error: Error message.
        """
        self.status = AgentStatus.ERROR
        self.last_error = error

    def clear_error(self) -> None:
        """Clear error state."""
        self.last_error = None
        if self.status == AgentStatus.ERROR:
            self.status = AgentStatus.IDLE

    def is_busy(self) -> bool:
        """Check if agent is currently busy.
        
        Returns:
            True if agent is processing or speaking.
        """
        return self.status in (AgentStatus.PROCESSING, AgentStatus.SPEAKING)

    def get_uptime(self) -> float:
        """Get agent uptime in seconds.
        
        Returns:
            Seconds since agent started.
        """
        return (datetime.now() - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary.
        
        Returns:
            State as dictionary.
        """
        return {
            "status": self.status.value,
            "current_task": self.current_task,
            "has_audio_output": self.last_audio_output is not None,
            "last_error": self.last_error,
            "metadata": self.metadata,
            "uptime_seconds": self.get_uptime(),
        }
