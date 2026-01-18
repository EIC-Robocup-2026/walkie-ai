"""Speech-to-Text module with pluggable providers."""

from .base import STTProvider
from .microphone import Microphone, list_audio_devices, print_audio_devices
from .speech_to_text import STT

__all__ = ["STT", "STTProvider", "Microphone", "list_audio_devices", "print_audio_devices"]
