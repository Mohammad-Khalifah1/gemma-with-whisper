"""
Action handlers — one function per intent.

Each handler is the single integration point between the AI pipeline and the
physical device (ESP32).  Today every handler prints to stdout.  When MQTT
support is added, replace the print() call with mqtt_client.publish() — the
rest of the pipeline stays untouched.

MQTT integration plan
---------------------
1. Install: uv add paho-mqtt
2. Create an MqttClient wrapper (e.g. src/mqtt_client.py) that connects to the
   broker and exposes a publish(topic, payload) method.
3. Import the client here and replace each TODO block with the publish call.

Example future body for handle_light_on():
    mqtt_client.publish(TOPIC_LIGHT, json.dumps({"state": "ON"}))
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MQTT configuration placeholders
# Uncomment and fill in when adding real broker support.
# ---------------------------------------------------------------------------
# BROKER_HOST  = "192.168.1.x"
# BROKER_PORT  = 1883
# TOPIC_LIGHT  = "home/room/light"   # ESP32 subscribes to this topic


def handle_light_on() -> None:
    """Execute the 'turn light on' command.

    Current behaviour : prints a confirmation to stdout.
    Future behaviour  : publishes {"state": "ON"} to the MQTT light topic so
                        the ESP32 switches the relay on.
    """
    # TODO: mqtt_client.publish(TOPIC_LIGHT, '{"state": "ON"}')
    print("[ACTION] light_on  → LIGHT ON")


def handle_light_off() -> None:
    """Execute the 'turn light off' command.

    Current behaviour : prints a confirmation to stdout.
    Future behaviour  : publishes {"state": "OFF"} to the MQTT light topic so
                        the ESP32 switches the relay off.
    """
    # TODO: mqtt_client.publish(TOPIC_LIGHT, '{"state": "OFF"}')
    print("[ACTION] light_off → LIGHT OFF")


def handle_unknown() -> None:
    """Fallback executed when no intent could be identified.

    Current behaviour : prints a warning to stdout.
    Future behaviour  : optionally publish to a debug/status MQTT topic.
    """
    # TODO: mqtt_client.publish(TOPIC_STATUS, '{"error": "unknown_intent"}')
    print("[ACTION] unknown   → no command sent")


# ---------------------------------------------------------------------------
# Dispatch table — maps each intent string to its handler.
# Add new intents here without touching the pipeline or main.py.
# ---------------------------------------------------------------------------
HANDLERS: dict[str, callable] = {
    "light_on":  handle_light_on,
    "light_off": handle_light_off,
    "unknown":   handle_unknown,
}


def dispatch(intent: str) -> None:
    """Call the action handler that corresponds to *intent*.

    Args:
        intent: Intent string returned by the pipeline
                (e.g. "light_on", "light_off", "unknown").

    If the intent is not in HANDLERS, falls back to handle_unknown().
    """
    handler = HANDLERS.get(intent, handle_unknown)
    handler()
