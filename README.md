# Voice-Controlled Light Switch

An offline, bilingual (Arabic + English) voice assistant that controls a smart light via voice commands. The system runs entirely on-device — no internet connection required. It is designed for IoT integration with an ESP32 microcontroller over MQTT.

---

## How It Works

```
PCM audio file
      │
      ▼
┌─────────────┐
│ WhisperSTT  │  Speech-to-Text  (faster-whisper, CPU, int8)
└──────┬──────┘
       │  transcript text
       ▼
┌──────────────┐
│  fast_match  │  Keyword lookup — instant, no LLM
└──────┬───────┘
       │ matched?
    ┌──┴──┐
   YES    NO
    │      │
    │      ▼
    │  ┌──────────────────┐
    │  │ GemmaClassifier  │  LLM fallback  (Gemma 4 2B, CPU)
    │  └────────┬─────────┘
    │           │
    └─────┬─────┘
          │  intent  (light_on | light_off | unknown)
          ▼
    ┌──────────┐
    │ dispatch │  Calls the action handler for the intent
    └──────────┘
          │
          ▼
   print to stdout        ← current behaviour
   mqtt_client.publish()  ← future behaviour (ESP32)
```

**Two supported intents:**

| Intent | Arabic examples | English examples |
|---|---|---|
| `light_on` | شغل الضوء، اضوي، اضئ، اشعل | turn on, switch on, light on |
| `light_off` | اطفئ الضوء، اطفي، اغلق، اقفل، سكر | turn off, switch off, light off |

---

## Project Structure

```
gemma/
├── main.py                      # Entry point — CLI and future MQTT listener
├── src/
│   ├── __init__.py
│   ├── whisper_stt.py           # WhisperSTT class  →  TranscriptResult
│   ├── gemma_classifier.py      # GemmaClassifier class  →  IntentResult
│   ├── intent_router.py         # fast_match(), get_command(), STORED_VALUES
│   └── actions.py               # Action handlers + dispatch table
├── tests/
│   ├── convert_to_pcm.py        # Audio → PCM converter (test utility)
│   ├── test_whisper_stt.py      # WhisperSTT unit + integration tests
│   ├── test_gemma_classifier.py # GemmaClassifier unit + integration tests
│   ├── test_intent_router.py    # fast_match / get_command tests
│   └── test_actions.py          # Action handler / dispatch tests
├── voice_samples/
│   ├── *.wav                    # Raw audio samples
│   └── pcm/                     # Converted PCM files (generated)
├── models/
│   ├── whisper/                 # Local Whisper model (not in git)
│   └── gemma4_2b/               # Local Gemma 4 2B model (not in git)
├── conftest.py                  # Pytest configuration
└── pyproject.toml
```

---

## Requirements

- Python ≥ 3.11
- [uv](https://github.com/astral-sh/uv) package manager
- ffmpeg (for audio conversion)
- Local model files at `./models/whisper` and `./models/gemma4_2b`

Install dependencies:

```bash
uv sync
```

Install ffmpeg (Ubuntu / Debian):

```bash
sudo apt install ffmpeg
```

---

## Running

### Single audio file

```bash
uv run python main.py --pcm voice_samples/pcm/speech.pcm
```

```bash
# Force a specific language (ar or en). Default: auto-detect.
uv run python main.py --pcm voice_samples/pcm/speech.pcm --language ar
```

```bash
# Override model paths
uv run python main.py \
  --pcm voice_samples/pcm/speech.pcm \
  --whisper-model ./models/whisper \
  --gemma-model   ./models/gemma4_2b
```

**All CLI arguments:**

| Argument | Default | Description |
|---|---|---|
| `--pcm` | *(required)* | Path to raw PCM file |
| `--sample-rate` | `16000` | PCM sample rate in Hz |
| `--channels` | `1` | Number of audio channels |
| `--language` | auto-detect | BCP-47 language hint (`ar`, `en`, …) |
| `--whisper-model` | `./models/whisper` | Local Whisper model path |
| `--gemma-model` | `./models/gemma4_2b` | Local Gemma 4 model path |

### Run all voice samples

```bash
for f in voice_samples/pcm/*.pcm; do
    echo "=== $f ==="
    uv run python main.py --pcm "$f"
done
```

---

## Preparing Audio Samples

Convert any audio format (WAV, OGG, MP3, FLAC, …) to the raw PCM format expected by the pipeline:

```bash
# Convert everything in voice_samples/ → voice_samples/pcm/
uv run python tests/convert_to_pcm.py

# Custom source and destination
uv run python tests/convert_to_pcm.py --src path/to/audio --dst path/to/pcm

# Custom sample rate
uv run python tests/convert_to_pcm.py --rate 16000
```

The converter resamples all files to **16 000 Hz, mono, int16 PCM** — the format expected by Whisper.

---

## Testing

### Unit tests (no models required — fast)

```bash
uv run python -m pytest tests/ -v -k "not integration"
```

### Integration tests (require model files)

```bash
# Whisper STT — transcribes all 4 voice samples and checks text + intent
uv run python -m pytest tests/test_whisper_stt.py -v -m integration

# Gemma classifier — runs real inference on Arabic phrases
uv run python -m pytest tests/test_gemma_classifier.py -v -m integration
```

### All tests

```bash
uv run python -m pytest tests/ -v
```

### Test coverage by module

| Test file | What it covers | Needs model? |
|---|---|---|
| `test_whisper_stt.py` | PCM→WAV conversion, validation, deduplication, transcription | Integration only |
| `test_gemma_classifier.py` | JSON parsing, prompt building, classify() with mocked model | Integration only |
| `test_intent_router.py` | fast_match() Arabic + English keywords, get_command(), registry structure | No |
| `test_actions.py` | Each handler output, dispatch() routing, HANDLERS registry | No |

---

## MQTT Integration

The system is architected to make MQTT integration a drop-in change. No modifications to the pipeline or `main.py` are needed — only two files need to be updated.

### Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              MQTT Broker                │
                    │         (e.g. Mosquitto on RPi)         │
                    └──────────┬──────────────────────────────┘
                               │
              ┌────────────────┼─────────────────────┐
              │                │                     │
              ▼                │                     ▼
   ┌──────────────────┐        │          ┌──────────────────┐
   │  ESP32 (device)  │        │          │  This system     │
   │                  │        │          │  (voice AI)      │
   │  subscribes to   │        │          │                  │
   │  home/light/cmd  │        │          │  publishes to    │
   │                  │        │          │  home/light/cmd  │
   │  publishes to    │        │          │                  │
   │  home/voice/pcm  │────────┘          │  subscribes to   │
   └──────────────────┘                   │  home/voice/pcm  │
                                          └──────────────────┘
```

### Step 1 — Create `src/mqtt_client.py`

```python
import paho.mqtt.client as mqtt
import json

BROKER_HOST  = "192.168.1.x"   # your broker IP
BROKER_PORT  = 1883
TOPIC_LIGHT  = "home/light/cmd"
TOPIC_VOICE  = "home/voice/pcm"

client = mqtt.Client()
client.connect(BROKER_HOST, BROKER_PORT)

def publish_light(state: str) -> None:
    """state: 'ON' or 'OFF'"""
    client.publish(TOPIC_LIGHT, json.dumps({"state": state}))
```

### Step 2 — Update `src/actions.py`

Replace the `print()` calls with MQTT publishes:

```python
# Before (current)
def handle_light_on() -> None:
    print("[ACTION] light_on → LIGHT ON")

# After (MQTT)
from src.mqtt_client import publish_light

def handle_light_on() -> None:
    publish_light("ON")

def handle_light_off() -> None:
    publish_light("OFF")
```

### Step 3 — Add MQTT subscriber in `main.py`

Wire the MQTT audio topic to `process_audio()`:

```python
import tempfile, os
from src.mqtt_client import client, TOPIC_VOICE
from main import process_audio

def on_voice_message(mqtt_client, userdata, msg):
    # Write the received PCM bytes to a temp file
    with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as f:
        f.write(msg.payload)
        tmp_path = f.name
    try:
        process_audio(tmp_path, language="ar")
    finally:
        os.unlink(tmp_path)

client.subscribe(TOPIC_VOICE)
client.on_message = on_voice_message
client.loop_forever()
```

### Install paho-mqtt when ready

```bash
uv add paho-mqtt
```

### MQTT topic convention

| Topic | Direction | Payload | Description |
|---|---|---|---|
| `home/voice/pcm` | ESP32 → AI | raw PCM bytes (int16, 16 kHz, mono) | Voice recording from ESP32 microphone |
| `home/light/cmd` | AI → ESP32 | `{"state": "ON"}` / `{"state": "OFF"}` | Light command for ESP32 relay |

---

## STT Accuracy Notes

Whisper is tuned for short voice commands with the following settings:

| Setting | Value | Reason |
|---|---|---|
| `language` | auto-detect | Avoids misreading English commands as Arabic when language is forced |
| `initial_prompt` | Smart-home phrases (AR + EN) | Primes the decoder toward expected vocabulary |
| `hotwords` | Key command words | Boosts scores for light-control vocabulary during beam search |
| `condition_on_previous_text` | `False` | Prevents hallucination propagation across segments |
| `temperature` | `0.0` | Fully deterministic; skips the noisy temperature-retry ladder |
| `hallucination_silence_threshold` | `0.5 s` | Drops repeated tokens generated during silent regions |
| `_deduplicate()` | post-process | Collapses any remaining consecutive repeated words |
