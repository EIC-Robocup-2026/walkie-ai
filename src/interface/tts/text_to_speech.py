"""Text-to-Speech module with configurable providers.

Usage:
    from src.interface.tts import TTS
    
    # Uses provider from config/settings.yaml
    tts = TTS()
    
    # Synthesize audio
    audio = await tts.synthesize("Hello, world!")
    
    # Stream audio
    async for chunk in tts.synthesize_stream("Hello, world!"):
        # process chunk
        pass
"""

from pathlib import Path
from typing import Any, AsyncIterator

import yaml

from .base import TTSProvider
from .providers import get_provider, list_providers


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load TTS configuration from settings.yaml.
    
    Args:
        config_path: Optional path to settings.yaml. 
                     Defaults to config/settings.yaml in project root.
    
    Returns:
        TTS configuration dictionary.
    """
    if config_path is None:
        # Find project root (where config/ directory is)
        current = Path(__file__).resolve()
        for parent in current.parents:
            settings_path = parent / "config" / "settings.yaml"
            if settings_path.exists():
                config_path = settings_path
                break
        else:
            raise FileNotFoundError("Could not find config/settings.yaml")
    
    with open(config_path) as f:
        settings = yaml.safe_load(f)
    
    return settings.get("tts", {})


class TTS:
    """Text-to-Speech interface with configurable providers."""

    def __init__(
        self,
        provider_name: str | None = None,
        config: dict[str, Any] | None = None,
        config_path: Path | None = None,
    ) -> None:
        """Initialize TTS with a provider.
        
        Args:
            provider_name: Override provider name (defaults to config).
            config: Override provider config (defaults to loading from file).
            config_path: Path to settings.yaml (defaults to config/settings.yaml).
        """
        # Load config if not provided
        if config is None:
            full_config = load_config(config_path)
            provider_name = provider_name or full_config.get("provider", "elevenlabs")
            config = full_config.get(provider_name, {})
        
        if provider_name is None:
            provider_name = "elevenlabs"
        
        self._provider_name = provider_name
        self._provider: TTSProvider = get_provider(provider_name, config)

    @property
    def provider_name(self) -> str:
        """Get the current provider name."""
        return self._provider_name

    @property
    def provider(self) -> TTSProvider:
        """Get the underlying provider instance."""
        return self._provider

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio.
        
        Args:
            text: The text to convert to speech.
            
        Returns:
            Audio data as bytes.
        """
        return await self._provider.synthesize(text)

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """Synthesize text to audio with streaming.
        
        Args:
            text: The text to convert to speech.
            
        Yields:
            Chunks of audio data as bytes.
        """
        async for chunk in self._provider.synthesize_stream(text):
            yield chunk

    def get_supported_formats(self) -> list[str]:
        """Get list of supported output formats for current provider."""
        return self._provider.get_supported_formats()

    @staticmethod
    def available_providers() -> list[str]:
        """List all available TTS providers."""
        return list_providers()
