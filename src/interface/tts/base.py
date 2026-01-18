"""Base class for Text-to-Speech providers."""

from abc import ABC, abstractmethod
from typing import AsyncIterator


class TTSProvider(ABC):
    """Abstract base class for TTS providers.
    
    Implement this class to add a new TTS provider.
    """

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio.
        
        Args:
            text: The text to convert to speech.
            
        Returns:
            Audio data as bytes.
        """
        pass

    @abstractmethod
    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """Synthesize text to audio with streaming.
        
        Args:
            text: The text to convert to speech.
            
        Yields:
            Chunks of audio data as bytes.
        """
        pass

    @abstractmethod
    def get_supported_formats(self) -> list[str]:
        """Get list of supported output formats.
        
        Returns:
            List of format identifiers (e.g., ['mp3', 'wav', 'pcm']).
        """
        pass
