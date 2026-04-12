# Gemma + Whisper Voice Control

This project converts voice (PCM) → text → intent using:
- faster-whisper (STT)
- Gemma 4 (LLM)

## Usage

```bash
uv run main.py --pcm sample.pcm --language ar