"""
Entry point: audio → STT → intent → action.

Pipeline
--------
1. WhisperSTT.transcribe()     — PCM audio bytes → transcript text
2. fast_match()                — keyword lookup (no LLM, instant)
3. GemmaClassifier.classify()  — LLM fallback (only when fast_match misses)
4. dispatch()                  — calls the action handler for the detected intent

IoT / MQTT integration plan
----------------------------
The function process_audio() below is the single entry point for the pipeline.
When MQTT support is added:

  - An MQTT subscriber receives a raw PCM payload on a topic like
    "home/voice/input" and writes it to a temporary file, then calls
    process_audio() with that path.
  - The action handlers in src/actions.py will publish the result back to the
    broker instead of printing to stdout.

No changes to this file or the pipeline will be needed — only src/actions.py
and a new src/mqtt_client.py need to be updated.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from src import GemmaClassifier, WhisperSTT, dispatch, fast_match

DEFAULT_WHISPER_MODEL = "./models/whisper"
DEFAULT_GEMMA_MODEL   = "./models/gemma4_2b"


# ---------------------------------------------------------------------------
# Core pipeline — callable from CLI, MQTT subscriber, or unit tests
# ---------------------------------------------------------------------------

def process_audio(
    pcm_path: str | Path,
    sample_rate: int = 16000,
    channels: int = 1,
    language: Optional[str] = None,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    gemma_model: str = DEFAULT_GEMMA_MODEL,
) -> str:
    """Run the full voice-to-action pipeline on a PCM audio file.

    Steps:
        1. Transcribe the audio with Whisper.
        2. Try fast keyword matching.
        3. Fall back to Gemma LLM if no keyword matched.
        4. Dispatch the detected intent to its action handler.

    Args:
        pcm_path:      Path to the raw PCM file (int16, little-endian).
        sample_rate:   PCM sample rate in Hz (default 16 000).
        channels:      Number of audio channels (default 1 = mono).
        language:      BCP-47 hint for Whisper ("ar", "en", or None for auto).
        whisper_model: Local path to the Whisper model directory.
        gemma_model:   Local path to the Gemma 4 model directory.

    Returns:
        The detected intent string (e.g. "light_on", "light_off", "unknown").

    # Future MQTT usage example:
    #   def on_mqtt_message(client, userdata, msg):
    #       tmp = write_to_tempfile(msg.payload)
    #       process_audio(tmp, language="ar")
    """
    # Step 1 — Speech to text
    stt = WhisperSTT(model_path=whisper_model)
    transcript = stt.transcribe(
        pcm_path=pcm_path,
        sample_rate=sample_rate,
        channels=channels,
        language=language,
    )

    print(f"[STT] language={transcript.language}  prob={transcript.language_probability:.2f}"
          if transcript.language_probability is not None
          else f"[STT] language={transcript.language}")
    print(f"[STT] text={transcript.text!r}")

    if not transcript.text:
        print("[PIPELINE] empty transcript")
        dispatch("unknown")
        return "unknown"
    # If we need using model confidence to LLM, by replacing 2,and 3 with the next edit  .
        # Step 2 — LLM matching (always used)
        # clf = GemmaClassifier(model_path=gemma_model)
        # result = clf.classify(transcript.text)
        # intent = result.intent
        # print(f"[GEMMA]  intent={intent}  confidence={result.confidence:.2f}")


    # Step 2 — Fast keyword match
    intent = fast_match(transcript.text)

    if intent is not None:
        print(f"[FAST]   intent={intent}  (keyword match, no LLM)")
    else:
        # Step 3 — LLM fallback
        clf = GemmaClassifier(model_path=gemma_model)
        result = clf.classify(transcript.text)
        intent = result.intent
        print(f"[GEMMA]  intent={intent}  confidence={result.confidence:.2f}")

    # Step 4 — Execute action
    dispatch(intent)

    return intent


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Voice-controlled light switch — PCM → STT → intent → action"
    )
    parser.add_argument("--pcm",          required=True,  help="Path to raw PCM file")
    parser.add_argument("--sample-rate",  type=int, default=16000)
    parser.add_argument("--channels",     type=int, default=1)
    parser.add_argument("--language",     default=None, help="ar | en | (auto-detect)")
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--gemma-model",   default=DEFAULT_GEMMA_MODEL)
    args = parser.parse_args()

    process_audio(
        pcm_path=args.pcm,
        sample_rate=args.sample_rate,
        channels=args.channels,
        language=args.language,
        whisper_model=args.whisper_model,
        gemma_model=args.gemma_model,
    )


if __name__ == "__main__":
    main()
