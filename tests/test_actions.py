"""
Tests for src/actions.py

Run with:
    uv run python -m pytest tests/test_actions.py -v
"""

import pytest
from src.actions import (
    dispatch,
    handle_light_on,
    handle_light_off,
    handle_unknown,
    HANDLERS,
)


# ---------------------------------------------------------------------------
# Individual handler output
# ---------------------------------------------------------------------------

class TestHandlers:
    def test_handle_light_on_prints(self, capsys):
        handle_light_on()
        out = capsys.readouterr().out
        assert "light_on" in out.lower() or "light on" in out.lower()

    def test_handle_light_off_prints(self, capsys):
        handle_light_off()
        out = capsys.readouterr().out
        assert "light_off" in out.lower() or "light off" in out.lower()

    def test_handle_unknown_prints(self, capsys):
        handle_unknown()
        out = capsys.readouterr().out
        assert "unknown" in out.lower()


# ---------------------------------------------------------------------------
# dispatch() routing
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_dispatch_light_on_calls_correct_handler(self, capsys):
        dispatch("light_on")
        out = capsys.readouterr().out
        assert "LIGHT ON" in out

    def test_dispatch_light_off_calls_correct_handler(self, capsys):
        dispatch("light_off")
        out = capsys.readouterr().out
        assert "LIGHT OFF" in out

    def test_dispatch_unknown_calls_unknown_handler(self, capsys):
        dispatch("unknown")
        out = capsys.readouterr().out
        assert "unknown" in out.lower()

    def test_dispatch_unregistered_intent_falls_back_to_unknown(self, capsys):
        dispatch("start_conveyor")   # not in HANDLERS
        out = capsys.readouterr().out
        assert "unknown" in out.lower()

    def test_dispatch_empty_string_falls_back_to_unknown(self, capsys):
        dispatch("")
        out = capsys.readouterr().out
        assert "unknown" in out.lower()


# ---------------------------------------------------------------------------
# HANDLERS registry structure
# ---------------------------------------------------------------------------

class TestHandlersRegistry:
    def test_light_on_registered(self):
        assert "light_on" in HANDLERS

    def test_light_off_registered(self):
        assert "light_off" in HANDLERS

    def test_unknown_registered(self):
        assert "unknown" in HANDLERS

    def test_all_handlers_are_callable(self):
        for intent, handler in HANDLERS.items():
            assert callable(handler), f"Handler for '{intent}' is not callable"
