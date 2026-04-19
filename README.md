# Voice-Controlled Light Switch

An offline, bilingual (Arabic + English) voice assistant that controls a smart light via voice commands. The system runs entirely on-device — no internet connection required. It is designed for IoT integration with an ESP32 microcontroller over MQTT.

---

## How It Works

The pipeline supports three modes selectable via `--mode` at runtime:

### Mode A — Gemma only (current, maximum accuracy)

Every transcript is classified by the LLM regardless of how obvious the command is.
Best when accuracy matters more than response time.

```
PCM audio file
      │
      ▼
┌─────────────┐
│ WhisperSTT  │  Speech-to-Text  (faster-whisper, CPU, int8)
└──────┬──────┘
       │  transcript text
       ▼
┌──────────────────┐
│ GemmaClassifier  │  LLM intent classification  (Gemma 4 2B, CPU)
└────────┬─────────┘
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

### Mode B — fast_match first, Gemma as fallback (faster response)

Common commands are matched instantly via keyword lookup.
Gemma is only loaded when no keyword matches.
Best for IoT real-time control where commands are predictable.

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

To switch to Mode B, replace the classification block in `main.py`:

```python
# Mode B — uncomment this block and remove the Gemma-only block below it
intent = fast_match(transcript.text)
if intent is not None:
    print(f"[FAST]  intent={intent}  (keyword match, no LLM)")
else:
    clf = GemmaClassifier(model_path=gemma_model)
    result = clf.classify(transcript.text)
    intent = result.intent
    print(f"[GEMMA] intent={intent}  confidence={result.confidence:.2f}")
```

### Mode C — MQTT (production IoT mode)

The system connects to an MQTT broker, listens for raw PCM audio published by
the ESP32 on `home/voice/pcm`, runs the full pipeline, then publishes the light
command back to the ESP32 on `home/light/cmd`.  No PCM file path is needed —
audio arrives over the network in real time.

```
ESP32 microphone
      │  raw PCM bytes  →  home/voice/pcm
      ▼
┌──────────────┐
│  MQTT Broker │  (e.g. Mosquitto on Raspberry Pi)
└──────┬───────┘
       │  home/voice/pcm payload
       ▼
┌──────────────────────────────────────┐
│  MQTTClient.start_voice_listener()   │  writes PCM to temp file
└──────────────┬───────────────────────┘
               │  temp PCM path
               ▼
        process_audio()               ← same pipeline as Mode A / B
               │  intent
               ▼
        dispatch(intent)
               │
               ▼
┌──────────────────────────────────────┐
│  MQTTClient.publish_light("ON/OFF")  │  home/light/cmd → ESP32 relay
└──────────────────────────────────────┘
```

```bash
# Start MQTT mode
uv run python main.py --mode mqtt --broker 192.168.1.100

# Optional: override port and models
uv run python main.py --mode mqtt --broker 192.168.1.100 --port 1883 \
    --whisper-model ./models/whisper --gemma-model ./models/gemma4_2b
```

**MQTT topic convention:**

| Topic | Direction | Payload |
|---|---|---|
| `home/voice/pcm` | ESP32 → AI | Raw PCM bytes (int16, 16 kHz, mono) |
| `home/light/cmd` | AI → ESP32 | `{"state": "ON"}` / `{"state": "OFF"}` |
| `home/status` | AI → Broker | `{"status": "OK/ERROR", "intent": "..."}` |

**Two supported intents:**

| Intent | Arabic examples | English examples |
|---|---|---|
| `light_on` | شغل الضوء، اضوي، اضئ، اشعل | turn on, switch on, light on |
| `light_off` | اطفئ الضوء، اطفي، اغلق، اقفل، سكر | turn off, switch off, light off |

---

## Project Structure

```
gemma/
├── main.py                      # Entry point — cli mode and mqtt mode
├── src/
│   ├── __init__.py
│   ├── whisper_stt.py           # WhisperSTT class  →  TranscriptResult
│   ├── gemma_classifier.py      # GemmaClassifier class  →  IntentResult
│   ├── intent_router.py         # fast_match(), get_command(), STORED_VALUES
│   ├── actions.py               # Action handlers + dispatch + configure_mqtt()
│   └── mqtt_client.py           # MQTTClient — connect, publish, voice listener
├── tests/
│   ├── convert_to_pcm.py        # Audio → PCM converter (test utility)
│   ├── test_whisper_stt.py      # WhisperSTT unit + integration tests
│   ├── test_gemma_classifier.py # GemmaClassifier unit + integration tests
│   ├── test_intent_router.py    # fast_match / get_command tests
│   ├── test_actions.py          # Action handler / dispatch tests
│   └── test_mqtt_client.py      # MQTTClient + MQTT transport tests
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

### Mode A / B — CLI (pass a PCM file)

```bash
# Basic
uv run python main.py --pcm voice_samples/pcm/speech.pcm

# Force a specific language (ar or en). Default: auto-detect.
uv run python main.py --pcm voice_samples/pcm/speech.pcm --language ar

# Override model paths
uv run python main.py \
  --pcm voice_samples/pcm/speech.pcm \
  --whisper-model ./models/whisper \
  --gemma-model   ./models/gemma4_2b
```

### Mode C — MQTT (receive audio from ESP32)

```bash
# Connect to broker and start listening
uv run python main.py --mode mqtt --broker 192.168.1.100

# With custom port and models
uv run python main.py --mode mqtt --broker 192.168.1.100 --port 1883 \
  --whisper-model ./models/whisper --gemma-model ./models/gemma4_2b
```

**All arguments:**

| Argument | Default | Modes | Description |
|---|---|---|---|
| `--mode` | `cli` | all | `cli` or `mqtt` |
| `--pcm` | *(required in cli)* | cli | Path to raw PCM file |
| `--broker` | `192.168.1.100` | mqtt | MQTT broker IP address |
| `--port` | `1883` | mqtt | MQTT broker port |
| `--sample-rate` | `16000` | all | PCM sample rate in Hz |
| `--channels` | `1` | all | Number of audio channels |
| `--language` | auto-detect | all | BCP-47 hint (`ar`, `en`, …) |
| `--whisper-model` | `./models/whisper` | all | Local Whisper model path |
| `--gemma-model` | `./models/gemma4_2b` | all | Local Gemma 4 model path |

### Run all voice samples (CLI)

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

## ESP32 Integration Requirements

This section describes what the ESP32 firmware must implement to work with this system.

### Infrastructure

Install a MQTT broker (e.g. Mosquitto) on a Raspberry Pi or any Linux machine on the same network:

```bash
sudo apt install mosquitto mosquitto-clients
sudo systemctl enable --now mosquitto
```

### Audio publisher (ESP32 → AI)

The ESP32 must capture microphone audio and publish it to `home/voice/pcm`:

- Read microphone at **16000 Hz, mono, 16-bit signed int (little-endian)**
- Publish the **raw PCM bytes only** — no WAV header, no container
- Send a complete clip per message (not a stream) — triggered by a button press or silence detection

### Command subscriber (ESP32 → Relay)

The ESP32 must subscribe to `home/light/cmd` and control the relay:

- Subscribe to `home/light/cmd`
- Parse the JSON payload: `{"state": "ON", "timestamp": "2025-..."}`
- Read `state` only — ignore `timestamp`
- Toggle the relay: `"ON"` → relay closed, `"OFF"` → relay open

### PCM audio format

| Parameter | Value |
|---|---|
| Sample rate | 16000 Hz |
| Channels | 1 (mono) |
| Bit depth | 16-bit signed integer |
| Byte order | little-endian |
| Container | **none** — raw PCM bytes only, no WAV header |

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
