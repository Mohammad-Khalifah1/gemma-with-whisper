"""
Action handlers — one function per intent.

Two transport modes are supported:

  CLI mode (default)
      Each handler prints to stdout.  No external dependencies.

  MQTT mode
      Call configure_mqtt(client) once at startup.  After that every handler
      publishes to the broker instead of printing, so the ESP32 receives the
      command directly.  The rest of the pipeline is untouched.

Switching modes:
    from src.actions import configure_mqtt
    from src.mqtt_client import MQTTClient

    mqtt = MQTTClient(broker_host="192.168.1.x")
    mqtt.connect()
    configure_mqtt(mqtt)          # swap handlers to MQTT transport
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.mqtt_client import MQTTClient

# Active MQTT client — None means CLI / print mode.
_mqtt: Optional["MQTTClient"] = None


def configure_mqtt(client: "MQTTClient") -> None:
    """Switch all action handlers to MQTT transport.

    Args:
        client: A connected MQTTClient instance.
                Pass None to revert to CLI / print mode.
    """
    global _mqtt
    _mqtt = client


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def handle_light_on() -> None:
    """Execute the 'turn light on' command."""
    if _mqtt is not None:
        _mqtt.publish_light("ON")
    else:
        print("[ACTION] light_on  → LIGHT ON")


def handle_light_off() -> None:
    """Execute the 'turn light off' command."""
    if _mqtt is not None:
        _mqtt.publish_light("OFF")
    else:
        print("[ACTION] light_off → LIGHT OFF")


def handle_unknown() -> None:
    """Fallback when no intent could be identified."""
    if _mqtt is not None:
        _mqtt.publish_status("OK", intent="unknown")
    else:
        print("[ACTION] unknown   → no command sent")


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

HANDLERS: dict[str, callable] = {
    "light_on":  handle_light_on,
    "light_off": handle_light_off,
    "unknown":   handle_unknown,
}


def dispatch(intent: str) -> None:
    """Call the action handler for *intent*, falling back to handle_unknown().

    Args:
        intent: Intent string from the pipeline (e.g. "light_on").
    """
    handler = HANDLERS.get(intent, handle_unknown)
    handler()
