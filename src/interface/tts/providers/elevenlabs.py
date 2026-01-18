"""ElevenLabs TTS provider implementation."""

import os
from typing import Any, AsyncIterator

from elevenlabs.client import AsyncElevenLabs

from ..base import TTSProvider


class ElevenLabsProvider(TTSProvider):
    """ElevenLabs Text-to-Speech provider using official SDK."""

    SUPPORTED_FORMATS = [
        "mp3_44100_128",
        "mp3_22050_32",
        "pcm_16000",
        "pcm_22050",
        "pcm_24000",
        "pcm_44100",
    ]

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize ElevenLabs provider."""
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable is required")

        self.client = AsyncElevenLabs(api_key=api_key)
        self.voice_id = config.get("voice_id", "21m00Tcm4TlvDq8ikWAM")
        self.model_id = config.get("model_id", "eleven_multilingual_v2")
        self.output_format = config.get("output_format", "mp3_44100_128")

        self.voice_settings = {
            "stability": config.get("stability", 0.5),
            "similarity_boost": config.get("similarity_boost", 0.75),
            "style": config.get("style", 0.0),
            "use_speaker_boost": config.get("use_speaker_boost", True),
        }

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio."""
        audio = self.client.text_to_speech.convert(
            text=text,
            voice_id=self.voice_id,
            model_id=self.model_id,
            output_format=self.output_format,
            voice_settings=self.voice_settings,
        )
        return b"".join([chunk async for chunk in audio])

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """Synthesize text to audio with streaming."""
        async for chunk in self.client.text_to_speech.convert(
            text=text,
            voice_id=self.voice_id,
            model_id=self.model_id,
            output_format=self.output_format,
            voice_settings=self.voice_settings,
        ):
            yield chunk

    def get_supported_formats(self) -> list[str]:
        """Get list of supported output formats."""
        return self.SUPPORTED_FORMATS.copy()
