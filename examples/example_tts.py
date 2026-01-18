"""Example usage of the TTS module."""

import asyncio
from pathlib import Path

from src.interface.tts import TTS

from dotenv import load_dotenv
load_dotenv()

async def main():
    # Initialize TTS (uses provider from config/settings.yaml)
    tts = TTS()
    print(f"Using provider: {tts.provider_name}")
    print(f"Available providers: {TTS.available_providers()}")

    # Synthesize text to audio
    text = "Hello! This is a test of the text to speech module."
    print(f"\nSynthesizing: {text}")
    
    audio = await tts.synthesize(text)
    
    # Save to file
    output_path = Path("output.mp3")
    output_path.write_bytes(audio)
    print(f"Saved audio to {output_path} ({len(audio)} bytes)")

    # Example with streaming
    print("\nStreaming example:")
    stream_path = Path("output_stream.mp3")
    with open(stream_path, "wb") as f:
        async for chunk in tts.synthesize_stream(text):
            f.write(chunk)
            print(f"  Received chunk: {len(chunk)} bytes")
    print(f"Saved streamed audio to {stream_path}")


if __name__ == "__main__":
    asyncio.run(main())
