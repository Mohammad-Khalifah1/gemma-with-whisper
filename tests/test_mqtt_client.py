"""
Tests for src/mqtt_client.py and MQTT transport in src/actions.py

All tests mock the paho-mqtt client — no real broker needed.

Run with:
    uv run python -m pytest tests/test_mqtt_client.py -v
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, call, patch

import pytest

from src.mqtt_client import (
    MQTTClient,
    TOPIC_LIGHT,
    TOPIC_STATUS,
    TOPIC_VOICE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client() -> tuple[MQTTClient, MagicMock]:
    """Return an MQTTClient with its internal paho client mocked."""
    with patch("src.mqtt_client.mqtt.Client") as MockPaho:
        mock_paho = MockPaho.return_value
        mock_paho.publish.return_value = MagicMock(rc=0)
        client = MQTTClient(broker_host="127.0.0.1", broker_port=1883)
        client._client = mock_paho
        client._connected = True
    return client, mock_paho


# ---------------------------------------------------------------------------
# MQTTClient — initialization
# ---------------------------------------------------------------------------

class TestMQTTClientInit:
    def test_stores_broker_settings(self):
        with patch("src.mqtt_client.mqtt.Client"):
            c = MQTTClient(broker_host="10.0.0.1", broker_port=1884)
        assert c.broker_host == "10.0.0.1"
        assert c.broker_port == 1884

    def test_default_broker_host_from_env(self, monkeypatch):
        import importlib
        monkeypatch.setenv("MQTT_BROKER_HOST", "10.0.0.99")
        import src.mqtt_client as m
        importlib.reload(m)          # re-evaluate module-level os.getenv() call
        assert m.DEFAULT_BROKER_HOST == "10.0.0.99"
        importlib.reload(m)          # restore original state after test


# ---------------------------------------------------------------------------
# MQTTClient — publish_light
# ---------------------------------------------------------------------------

class TestPublishLight:
    def test_publish_on_returns_true(self):
        client, mock_paho = _make_client()
        result = client.publish_light("ON")
        assert result is True

    def test_publish_off_returns_true(self):
        client, mock_paho = _make_client()
        result = client.publish_light("OFF")
        assert result is True

    def test_publishes_to_correct_topic(self):
        client, mock_paho = _make_client()
        client.publish_light("ON")
        topic = mock_paho.publish.call_args[0][0]
        assert topic == TOPIC_LIGHT

    def test_payload_contains_state_on(self):
        client, mock_paho = _make_client()
        client.publish_light("ON")
        raw = mock_paho.publish.call_args[0][1]
        payload = json.loads(raw)
        assert payload["state"] == "ON"

    def test_payload_contains_state_off(self):
        client, mock_paho = _make_client()
        client.publish_light("OFF")
        raw = mock_paho.publish.call_args[0][1]
        payload = json.loads(raw)
        assert payload["state"] == "OFF"

    def test_payload_contains_timestamp(self):
        client, mock_paho = _make_client()
        client.publish_light("ON")
        raw = mock_paho.publish.call_args[0][1]
        payload = json.loads(raw)
        assert "timestamp" in payload

    def test_uses_qos_1(self):
        client, mock_paho = _make_client()
        client.publish_light("ON")
        kwargs = mock_paho.publish.call_args[1]
        assert kwargs.get("qos") == 1

    def test_returns_false_on_failure(self):
        client, mock_paho = _make_client()
        mock_paho.publish.return_value = MagicMock(rc=1)   # non-zero = error
        result = client.publish_light("ON")
        assert result is False


# ---------------------------------------------------------------------------
# MQTTClient — publish_status
# ---------------------------------------------------------------------------

class TestPublishStatus:
    def test_publishes_to_status_topic(self):
        client, mock_paho = _make_client()
        client.publish_status("OK", intent="light_on")
        topic = mock_paho.publish.call_args[0][0]
        assert topic == TOPIC_STATUS

    def test_payload_contains_status(self):
        client, mock_paho = _make_client()
        client.publish_status("ERROR", error="timeout")
        raw = mock_paho.publish.call_args[0][1]
        payload = json.loads(raw)
        assert payload["status"] == "ERROR"
        assert payload["error"] == "timeout"

    def test_payload_omits_empty_fields(self):
        client, mock_paho = _make_client()
        client.publish_status("OK")
        raw = mock_paho.publish.call_args[0][1]
        payload = json.loads(raw)
        assert "intent" not in payload
        assert "error" not in payload


# ---------------------------------------------------------------------------
# MQTTClient — voice listener
# ---------------------------------------------------------------------------

class TestVoiceListener:
    def test_subscribes_to_voice_topic(self):
        client, mock_paho = _make_client()
        client.start_voice_listener(on_audio=lambda p: None)
        mock_paho.subscribe.assert_called_once_with(TOPIC_VOICE, qos=1)

    def test_on_audio_called_with_temp_file(self):
        client, mock_paho = _make_client()
        received_paths = []

        client.start_voice_listener(on_audio=lambda p: received_paths.append(p))

        # Simulate MQTT message arrival
        fake_msg = MagicMock()
        fake_msg.topic = TOPIC_VOICE
        fake_msg.payload = b"\x00\x01" * 100
        client._client.on_message(None, None, fake_msg)

        assert len(received_paths) == 1
        # Temp file should be cleaned up after the callback
        assert not os.path.exists(received_paths[0])

    def test_wrong_topic_ignored(self):
        client, mock_paho = _make_client()
        called = []

        client.start_voice_listener(on_audio=lambda p: called.append(p))

        fake_msg = MagicMock()
        fake_msg.topic = "other/topic"
        fake_msg.payload = b"\x00\x01"
        client._client.on_message(None, None, fake_msg)

        assert called == []

    def test_pipeline_error_publishes_error_status(self):
        client, mock_paho = _make_client()

        def _broken(_path):
            raise RuntimeError("model crashed")

        client.start_voice_listener(on_audio=_broken)

        fake_msg = MagicMock()
        fake_msg.topic = TOPIC_VOICE
        fake_msg.payload = b"\x00\x01" * 10
        client._client.on_message(None, None, fake_msg)

        # Check that publish_status was called with ERROR
        calls = [json.loads(c[0][1]) for c in mock_paho.publish.call_args_list]
        statuses = [p["status"] for p in calls]
        assert "ERROR" in statuses


# ---------------------------------------------------------------------------
# MQTT transport in actions.py
# ---------------------------------------------------------------------------

class TestActionsWithMQTT:
    def setup_method(self):
        # Reset to CLI mode before each test
        import src.actions as a
        a._mqtt = None

    def test_cli_mode_prints_light_on(self, capsys):
        from src.actions import handle_light_on
        handle_light_on()
        assert "LIGHT ON" in capsys.readouterr().out

    def test_cli_mode_prints_light_off(self, capsys):
        from src.actions import handle_light_off
        handle_light_off()
        assert "LIGHT OFF" in capsys.readouterr().out

    def test_mqtt_mode_calls_publish_light_on(self):
        from src.actions import configure_mqtt, handle_light_on
        mock_client = MagicMock()
        configure_mqtt(mock_client)
        handle_light_on()
        mock_client.publish_light.assert_called_once_with("ON")

    def test_mqtt_mode_calls_publish_light_off(self):
        from src.actions import configure_mqtt, handle_light_off
        mock_client = MagicMock()
        configure_mqtt(mock_client)
        handle_light_off()
        mock_client.publish_light.assert_called_once_with("OFF")

    def test_mqtt_mode_does_not_print(self, capsys):
        from src.actions import configure_mqtt, handle_light_on
        configure_mqtt(MagicMock())
        handle_light_on()
        assert capsys.readouterr().out == ""

    def test_configure_mqtt_none_reverts_to_cli(self, capsys):
        from src.actions import configure_mqtt, handle_light_on
        configure_mqtt(MagicMock())
        configure_mqtt(None)
        handle_light_on()
        assert "LIGHT ON" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Topic constants
# ---------------------------------------------------------------------------

class TestTopicConstants:
    def test_voice_topic_defined(self):
        assert TOPIC_VOICE == "home/voice/pcm"

    def test_light_topic_defined(self):
        assert TOPIC_LIGHT == "home/light/cmd"

    def test_status_topic_defined(self):
        assert TOPIC_STATUS == "home/status"
