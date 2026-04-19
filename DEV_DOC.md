# Developer Documentation

Technical reference for contributors and integrators.

---

## Architecture

```
src/
├── whisper_stt.py       WhisperSTT          PCM file → TranscriptResult
├── gemma_classifier.py  GemmaClassifier     text     → IntentResult
├── intent_router.py     fast_match()        text     → intent string (no LLM)
│                        get_command()       intent   → command payload dict
├── actions.py           dispatch()          intent   → handler call
│                        configure_mqtt()    swap transport (print ↔ MQTT)
└── mqtt_client.py       MQTTClient          connect, publish, voice listener
```

`main.py` owns the pipeline — it wires the modules together and exposes two entry points:

| Entry point | Used by |
|---|---|
| `process_audio(pcm_path, ...)` | CLI mode, MQTT voice listener, unit tests |
| `main()` | `__main__`, argparse |

---

## Module Interfaces

### `WhisperSTT`

```python
from src.whisper_stt import WhisperSTT, TranscriptResult

stt = WhisperSTT(model_path="./models/whisper", device="cpu", compute_type="int8")

result: TranscriptResult = stt.transcribe(
    pcm_path="audio.pcm",   # Path | str — raw int16 PCM
    sample_rate=16000,
    channels=1,
    language=None,           # None = auto-detect, "ar", "en", …
)

result.text                  # str
result.language              # str | None
result.language_probability  # float | None
```

Internal helpers (usable standalone):

```python
WhisperSTT._pcm_to_wav(pcm_path, wav_path, sample_rate, channels)  # → None
WhisperSTT._deduplicate("اطفئ الضوء الضوء الضوء")                  # → "اطفئ الضوء"
```

---

### `GemmaClassifier`

```python
from src.gemma_classifier import GemmaClassifier, IntentResult

clf = GemmaClassifier(model_path="./models/gemma4_2b")
clf.load()   # optional pre-warm; classify() lazy-loads if skipped

result: IntentResult = clf.classify("شغل الضوء")

result.intent      # "light_on" | "light_off" | "unknown"
result.confidence  # float  0.0 – 1.0
result.raw         # dict — full JSON from the model
```

Allowed intents: `("light_on", "light_off", "unknown")` — defined in `ALLOWED_INTENTS`.

---

### `intent_router`

```python
from src.intent_router import fast_match, get_command, STORED_VALUES, ALLOWED_INTENTS

intent = fast_match("اطفئ الضوء")   # → "light_off" | None
cmd    = get_command("light_on")     # → {"cmd": "LIGHT_ON"} | None
```

---

### `actions`

```python
from src.actions import dispatch, configure_mqtt

dispatch("light_on")    # calls handle_light_on() — print or MQTT depending on mode

# Switch to MQTT transport
from src.mqtt_client import MQTTClient
mqtt = MQTTClient(broker_host="192.168.1.x")
mqtt.connect()
configure_mqtt(mqtt)    # handlers now publish instead of print

# Revert to CLI/print
configure_mqtt(None)
```

---

### `MQTTClient`

```python
from src.mqtt_client import MQTTClient

client = MQTTClient(broker_host="192.168.1.x", broker_port=1883)
client.connect()

client.publish_light("ON")   # → bool  (True = queued OK)
client.publish_light("OFF")

client.publish_status("OK", intent="light_on")
client.publish_status("ERROR", error="model failed")

client.start_voice_listener(on_audio=process_audio)  # non-blocking subscribe
client.loop_forever()   # blocks until Ctrl-C, then disconnects
client.disconnect()
```

Environment variable overrides:

```bash
export MQTT_BROKER_HOST=192.168.1.x
export MQTT_BROKER_PORT=1883
```

---

## MQTT Integration — Step by Step

### Step 1 — `src/mqtt_client.py`  *(already implemented)*

```python
import paho.mqtt.client as mqtt
import json

BROKER_HOST = "192.168.1.x"   # your broker IP
BROKER_PORT = 1883
TOPIC_LIGHT = "home/light/cmd"
TOPIC_VOICE = "home/voice/pcm"

client = mqtt.Client()
client.connect(BROKER_HOST, BROKER_PORT)

def publish_light(state: str) -> None:
    """state: 'ON' or 'OFF'"""
    client.publish(TOPIC_LIGHT, json.dumps({"state": state}))
```

### Step 2 — `src/actions.py`  *(already implemented)*

Switch from `print()` to MQTT publish by calling `configure_mqtt()`:

```python
# Before (CLI / print mode)
def handle_light_on() -> None:
    print("[ACTION] light_on → LIGHT ON")

# After (MQTT mode) — achieved by calling configure_mqtt(client)
from src.mqtt_client import publish_light

def handle_light_on() -> None:
    publish_light("ON")

def handle_light_off() -> None:
    publish_light("OFF")
```

### Step 3 — MQTT subscriber in `main.py`  *(already implemented)*

`process_audio()` is called from the MQTT voice listener callback:

```python
import tempfile, os
from src.mqtt_client import MQTTClient
from main import process_audio

def on_voice_message(mqtt_client, userdata, msg):
    with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as f:
        f.write(msg.payload)
        tmp_path = f.name
    try:
        process_audio(tmp_path, language="ar")
    finally:
        os.unlink(tmp_path)

client = MQTTClient(broker_host="192.168.1.x")
client.connect()
client.start_voice_listener(on_audio=process_audio)
client.loop_forever()
```

Run it with:

```bash
uv run python main.py --mode mqtt --broker 192.168.1.x
```

### Install paho-mqtt

```bash
uv add paho-mqtt
```

---

## MQTT Topic Convention

| Topic | Direction | Payload | Description |
|---|---|---|---|
| `home/voice/pcm` | ESP32 → AI | Raw PCM bytes (int16, 16 kHz, mono) | Voice recording from ESP32 microphone |
| `home/light/cmd` | AI → ESP32 | `{"state": "ON", "timestamp": "…"}` | Light command for ESP32 relay |
| `home/status` | AI → Broker | `{"status": "OK\|ERROR", "intent": "…"}` | Pipeline status for monitoring |

---

## STT Accuracy Settings

| Parameter | Value | Reason |
|---|---|---|
| `language` | `None` (auto) | Prevents misreading English as Arabic when language is forced |
| `initial_prompt` | Smart-home phrases AR + EN | Primes decoder toward expected vocabulary |
| `hotwords` | Key command words | Boosts scores during beam search |
| `condition_on_previous_text` | `False` | Prevents hallucination propagation across segments |
| `temperature` | `0.0` | Deterministic; skips the noisy retry ladder |
| `hallucination_silence_threshold` | `0.5 s` | Drops repeated tokens in silent regions |
| `_deduplicate()` | post-process | Collapses any remaining consecutive duplicate words |

---

## Adding a New Intent

1. **`src/intent_router.py`** — add entry to `STORED_VALUES` and `_KEYWORDS`:

```python
STORED_VALUES["fan_on"] = {"cmd": "FAN_ON"}

_KEYWORDS["fan_on"] = ("turn on fan", "fan on", "شغل المروحة", "شغل الفان")
```

2. **`src/gemma_classifier.py`** — add to `ALLOWED_INTENTS`:

```python
ALLOWED_INTENTS: tuple[str, ...] = ("light_on", "light_off", "fan_on", "unknown")
```

3. **`src/actions.py`** — add handler and register in `HANDLERS`:

```python
def handle_fan_on() -> None:
    if _mqtt is not None:
        _mqtt.publish_light("FAN_ON")   # or a dedicated publish method
    else:
        print("[ACTION] fan_on → FAN ON")

HANDLERS["fan_on"] = handle_fan_on
```

No changes needed in `main.py` or the pipeline.

---

## Running Tests

```bash
# Unit tests — no models or broker needed (~1.5 s)
uv run python -m pytest tests/ -v -k "not integration"

# Whisper integration — requires ./models/whisper and voice_samples/pcm/
uv run python -m pytest tests/test_whisper_stt.py -v -m integration

# Gemma integration — requires ./models/gemma4_2b
uv run python -m pytest tests/test_gemma_classifier.py -v -m integration
```

| Test file | Covers | Needs model |
|---|---|---|
| `test_whisper_stt.py` | PCM→WAV, `_deduplicate`, transcription | integration only |
| `test_gemma_classifier.py` | JSON parsing, prompt building, `classify()` | integration only |
| `test_intent_router.py` | `fast_match` AR + EN, `get_command`, registry | no |
| `test_actions.py` | handlers, `dispatch`, `HANDLERS` | no |
| `test_mqtt_client.py` | publish, voice listener, transport switch | no |
