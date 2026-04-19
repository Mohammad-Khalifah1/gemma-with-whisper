"""
Entry point: audio → STT → intent → action.

Pipeline
--------
1. WhisperSTT.transcribe()     — PCM audio → transcript text
2. GemmaClassifier.classify()  — transcript → intent
3. dispatch()                  — intent → action handler

Two runtime modes
-----------------
cli  (default)
    Pass a PCM file path via --pcm.  Results are printed to stdout.
    Good for development and one-shot testing.

        uv run python main.py --pcm voice_samples/pcm/speech.pcm

mqtt
    Connects to an MQTT broker and listens for raw PCM audio published by an
    ESP32 on home/voice/pcm.  After classification, the light command is
    published back to the ESP32 on home/light/cmd.
    No --pcm argument needed — audio arrives over the network.

        uv run python main.py --mode mqtt --broker 192.168.1.100

MQTT integration note
---------------------
In MQTT mode, process_audio() is called from the MQTT message callback
inside MQTTClient.start_voice_listener().  The function signature is
identical in both modes — only the transport layer (print vs. MQTT publish)
changes, handled transparently by src/actions.py.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from src import GemmaClassifier, WhisperSTT, dispatch
from src.actions import configure_mqtt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_WHISPER_MODEL = "./models/whisper"
DEFAULT_GEMMA_MODEL   = "./models/gemma4_2b"


# ---------------------------------------------------------------------------
# Core pipeline — shared between CLI and MQTT modes
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
        2. Classify the transcript with Gemma.
        3. Dispatch the detected intent to its action handler.

    Args:
        pcm_path:      Path to the raw PCM file (int16, little-endian).
        sample_rate:   PCM sample rate in Hz (default 16 000).
        channels:      Number of audio channels (default 1 = mono).
        language:      BCP-47 hint for Whisper ("ar", "en", or None = auto).
        whisper_model: Local path to the Whisper model directory.
        gemma_model:   Local path to the Gemma 4 model directory.

    Returns:
        The detected intent string ("light_on", "light_off", or "unknown").
    """
    # Step 1 — Speech to text
    stt = WhisperSTT(model_path=whisper_model)
    transcript = stt.transcribe(
        pcm_path=pcm_path,
        sample_rate=sample_rate,
        channels=channels,
        language=language,
    )

    if transcript.language_probability is not None:
        logger.info("STT language=%s prob=%.2f", transcript.language, transcript.language_probability)
    else:
        logger.info("STT language=%s", transcript.language)
    logger.info("STT text=%r", transcript.text)

    if not transcript.text:
        logger.warning("Empty transcript — dispatching unknown.")
        dispatch("unknown")
        return "unknown"

    # Step 2 — LLM classification
    clf = GemmaClassifier(model_path=gemma_model)
    result = clf.classify(transcript.text)
    logger.info("Gemma intent=%s confidence=%.2f", result.intent, result.confidence)

    # Step 3 — Execute action (print or MQTT publish, depending on mode)
    dispatch(result.intent)

    return result.intent


# ---------------------------------------------------------------------------
# CLI mode
# ---------------------------------------------------------------------------

def _run_cli(args: argparse.Namespace) -> None:
    process_audio(
        pcm_path=args.pcm,
        sample_rate=args.sample_rate,
        channels=args.channels,
        language=args.language,
        whisper_model=args.whisper_model,
        gemma_model=args.gemma_model,
    )


# ---------------------------------------------------------------------------
# MQTT mode
# ---------------------------------------------------------------------------

def _run_mqtt(args: argparse.Namespace) -> None:
    from src.mqtt_client import MQTTClient

    client = MQTTClient(broker_host=args.broker, broker_port=args.port)
    client.connect()

    # Swap action handlers to MQTT transport (publish instead of print).
    configure_mqtt(client)

    def _on_audio(pcm_path: str) -> None:
        process_audio(
            pcm_path=pcm_path,
            sample_rate=args.sample_rate,
            channels=args.channels,
            language=args.language,
            whisper_model=args.whisper_model,
            gemma_model=args.gemma_model,
        )

    client.start_voice_listener(on_audio=_on_audio)
    client.loop_forever()   # blocks until Ctrl-C


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Voice-controlled light switch — PCM → STT → Gemma → action"
    )

    # Shared arguments
    parser.add_argument("--mode",         choices=["cli", "mqtt"], default="cli",
                        help="cli: pass a PCM file; mqtt: listen for audio over MQTT (default: cli)")
    parser.add_argument("--sample-rate",  type=int, default=16000)
    parser.add_argument("--channels",     type=int, default=1)
    parser.add_argument("--language",     default=None, help="ar | en | (auto-detect)")
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--gemma-model",   default=DEFAULT_GEMMA_MODEL)

    # CLI-mode arguments
    parser.add_argument("--pcm", default=None,
                        help="[cli mode] Path to raw PCM file")

    # MQTT-mode arguments
    parser.add_argument("--broker", default="192.168.1.100",
                        help="[mqtt mode] MQTT broker IP address (default: 192.168.1.100)")
    parser.add_argument("--port",   type=int, default=1883,
                        help="[mqtt mode] MQTT broker port (default: 1883)")

    args = parser.parse_args()

    if args.mode == "cli":
        if not args.pcm:
            parser.error("--pcm is required in cli mode")
        _run_cli(args)
    else:
        _run_mqtt(args)


if __name__ == "__main__":
    main()
