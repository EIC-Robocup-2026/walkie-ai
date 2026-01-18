"""Speech-to-Text module with configurable providers.

Usage:
    from src.interface.asr import STT
    
    # Uses provider from config/settings.yaml
    stt = STT()
    
    # Transcribe audio
    text = stt.transcribe(audio_bytes)
"""

from pathlib import Path
from typing import Any

import yaml

from .base import STTProvider
from .providers import get_provider, list_providers


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load STT configuration from settings.yaml."""
    if config_path is None:
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
    
    return settings.get("stt", {})


class STT:
    """Speech-to-Text interface with configurable providers."""

    def __init__(
        self,
        provider_name: str | None = None,
        config: dict[str, Any] | None = None,
        config_path: Path | None = None,
    ) -> None:
        """Initialize STT with a provider."""
        if config is None:
            full_config = load_config(config_path)
            provider_name = provider_name or full_config.get("provider", "google")
            config = full_config.get(provider_name, {})
        
        if provider_name is None:
            provider_name = "google"
        
        self._provider_name = provider_name
        self._provider: STTProvider = get_provider(provider_name, config)

    @property
    def provider_name(self) -> str:
        """Get the current provider name."""
        return self._provider_name

    @property
    def provider(self) -> STTProvider:
        """Get the underlying provider instance."""
        return self._provider

    def transcribe(self, audio_content: bytes) -> str:
        """Transcribe audio to text."""
        return self._provider.transcribe(audio_content)

    def get_supported_languages(self) -> list[str]:
        """Get list of supported languages for current provider."""
        return self._provider.get_supported_languages()

    @staticmethod
    def available_providers() -> list[str]:
        """List all available STT providers."""
        return list_providers()
