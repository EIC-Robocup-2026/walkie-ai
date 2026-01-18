"""Text-to-Speech module with pluggable providers."""

from .base import TTSProvider
from .text_to_speech import TTS

__all__ = ["TTS", "TTSProvider"]
