"""
MQTT client wrapper for IoT integration with ESP32.

Responsibilities:
- Connect to an MQTT broker (e.g. Mosquitto on Raspberry Pi).
- Publish light commands to the ESP32 on home/light/cmd.
- Subscribe to home/voice/pcm to receive raw PCM audio from the ESP32
  microphone, then invoke the voice pipeline on each received clip.

Topic convention:
    home/voice/pcm   — ESP32 → AI  : raw PCM bytes (int16, 16 kHz, mono)
    home/light/cmd   — AI  → ESP32 : JSON {"state": "ON" | "OFF"}
    home/status      — AI  → Broker: JSON {"status": "...", "intent": "..."}

Typical usage (MQTT mode):
    client = MQTTClient(broker_host="192.168.1.x")
    client.connect()
    client.start_voice_listener(on_audio=process_audio_bytes)
    client.loop_forever()          # blocks; Ctrl-C to stop
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Callable, Optional

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default broker settings — override via MQTTClient() arguments or env vars.
# ---------------------------------------------------------------------------
DEFAULT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "192.168.1.100")
DEFAULT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))

TOPIC_VOICE  = "home/voice/pcm"    # subscribe — receive audio from ESP32
TOPIC_LIGHT  = "home/light/cmd"    # publish   — send command to ESP32
TOPIC_STATUS = "home/status"       # publish   — pipeline status / debug


class MQTTClient:
    """Thin wrapper around paho-mqtt for this voice-control system."""

    def __init__(
        self,
        broker_host: str = DEFAULT_BROKER_HOST,
        broker_port: int = DEFAULT_BROKER_PORT,
        client_id: str = "voice-ai",
    ) -> None:
        self.broker_host = broker_host
        self.broker_port = broker_port

        self._client = mqtt.Client(
            client_id=client_id,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._connected = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to the broker (blocking until the ACK is received).

        Raises:
            ConnectionError: If the broker is unreachable.
        """
        logger.info("Connecting to MQTT broker %s:%s …", self.broker_host, self.broker_port)
        self._client.connect(self.broker_host, self.broker_port, keepalive=60)
        self._client.loop_start()      # background thread for network I/O

    def disconnect(self) -> None:
        """Gracefully disconnect from the broker."""
        self._client.loop_stop()
        self._client.disconnect()
        logger.info("Disconnected from MQTT broker.")

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish_light(self, state: str) -> bool:
        """Send a light ON/OFF command to the ESP32.

        Args:
            state: "ON" or "OFF".

        Returns:
            True if the message was queued successfully, False otherwise.
        """
        payload = json.dumps({
            "state": state,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        result = self._client.publish(TOPIC_LIGHT, payload, qos=1)
        ok = result.rc == mqtt.MQTT_ERR_SUCCESS
        if ok:
            logger.info("Published light=%s to %s", state, TOPIC_LIGHT)
        else:
            logger.error("Failed to publish light=%s (rc=%s)", state, result.rc)
        return ok

    def publish_status(self, status: str, intent: str = "", error: str = "") -> bool:
        """Publish a pipeline status message for monitoring / debugging.

        Args:
            status: One of "OK", "ERROR", "PROCESSING".
            intent: Detected intent string (optional).
            error:  Error description when status is "ERROR" (optional).

        Returns:
            True if queued successfully.
        """
        payload: dict = {"status": status}
        if intent:
            payload["intent"] = intent
        if error:
            payload["error"] = error

        result = self._client.publish(TOPIC_STATUS, json.dumps(payload), qos=0)
        return result.rc == mqtt.MQTT_ERR_SUCCESS

    # ------------------------------------------------------------------
    # Voice listener
    # ------------------------------------------------------------------

    def start_voice_listener(
        self,
        on_audio: Callable[[str], None],
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        """Subscribe to the voice PCM topic and call *on_audio* for each clip.

        The raw PCM bytes received from the ESP32 are written to a temporary
        file and the path is passed to *on_audio* (e.g. process_audio()).
        The temp file is deleted automatically after the callback returns.

        Args:
            on_audio:    Callable that accepts a PCM file path string.
                         Typically: main.process_audio
            sample_rate: Expected PCM sample rate (default 16 000 Hz).
            channels:    Expected number of channels (default 1 = mono).
        """
        def _on_message(client, userdata, msg):
            if msg.topic != TOPIC_VOICE:
                return

            logger.info("Received %d bytes of PCM audio from %s", len(msg.payload), msg.topic)
            self.publish_status("PROCESSING")

            tmp_path: Optional[str] = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as f:
                    f.write(msg.payload)
                    tmp_path = f.name

                on_audio(tmp_path)

            except Exception as exc:
                logger.exception("Pipeline error: %s", exc)
                self.publish_status("ERROR", error=str(exc))
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        self._client.on_message = _on_message
        self._client.subscribe(TOPIC_VOICE, qos=1)
        logger.info("Subscribed to %s — waiting for audio …", TOPIC_VOICE)

    def loop_forever(self) -> None:
        """Block and process MQTT events until interrupted (Ctrl-C)."""
        logger.info("MQTT voice listener running. Press Ctrl-C to stop.")
        try:
            self._client.loop_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down MQTT listener …")
        finally:
            self.disconnect()

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            self._connected = True
            logger.info("Connected to MQTT broker successfully.")
        else:
            logger.error("MQTT connection refused (reason_code=%s).", reason_code)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        self._connected = False
        if reason_code != 0:
            logger.warning("Unexpected MQTT disconnection (reason_code=%s).", reason_code)
