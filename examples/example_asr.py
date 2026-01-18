"""Example usage of the ASR module with microphone input."""

from dotenv import load_dotenv
load_dotenv()

from src.interface.asr import STT
from src.interface.asr.microphone import Microphone, print_audio_devices


def main():
    print_audio_devices()
    # Initialize STT (uses provider from config/settings.yaml)
    stt = STT()
    print(f"Using provider: {stt.provider_name}")
    print(f"Supported languages: {stt.get_supported_languages()[:5]}...")

    # Initialize microphone with VAD
    mic = Microphone(
        device=3,
        threshold=0.5,              # VAD sensitivity
        min_silence_duration_ms=500, # End after 500ms of silence
    )

    print("\n--- Recording with Voice Activity Detection ---")
    print("Speak into the microphone. Recording will stop after you stop speaking.")
    print("Listening...")

    # Record until silence is detected
    audio = mic.record_until_silence(timeout=30.0)
    print(f"Recorded {len(audio)} bytes of audio")

    if audio:
        # Transcribe the recorded audio
        print("\nTranscribing...")
        text = stt.transcribe(audio)
        print(f"Transcription: {text}")
    else:
        print("No audio recorded")


if __name__ == "__main__":
    main()
