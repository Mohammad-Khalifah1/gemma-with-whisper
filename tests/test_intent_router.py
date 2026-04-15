"""
Tests for src/intent_router.py

Run with:
    uv run python -m pytest tests/test_intent_router.py -v
"""

import pytest
from src.intent_router import fast_match, get_command, STORED_VALUES, ALLOWED_INTENTS


# ---------------------------------------------------------------------------
# fast_match — English keywords
# ---------------------------------------------------------------------------

class TestFastMatchEnglish:
    def test_turn_on(self):
        assert fast_match("turn on the light") == "light_on"

    def test_switch_on(self):
        assert fast_match("switch on") == "light_on"

    def test_light_on(self):
        assert fast_match("light on") == "light_on"

    def test_open_the_light(self):
        assert fast_match("open the light") == "light_on"

    def test_turn_off(self):
        assert fast_match("turn off the light") == "light_off"

    def test_switch_off(self):
        assert fast_match("switch off") == "light_off"

    def test_light_off(self):
        assert fast_match("light off") == "light_off"

    def test_close_the_light(self):
        assert fast_match("close the light") == "light_off"

    def test_case_insensitive_on(self):
        assert fast_match("TURN ON THE LIGHT") == "light_on"

    def test_case_insensitive_off(self):
        assert fast_match("TURN OFF THE LIGHT") == "light_off"


# ---------------------------------------------------------------------------
# fast_match — Arabic keywords
# ---------------------------------------------------------------------------

class TestFastMatchArabic:
    def test_shagghel(self):
        assert fast_match("شغل الضوء") == "light_on"

    def test_adhwi(self):
        assert fast_match("اضوي الضوء") == "light_on"

    def test_adhi(self):
        assert fast_match("اضئ الضو") == "light_on"

    def test_eshaal(self):
        assert fast_match("اشعل الضوء") == "light_on"

    def test_itfi(self):
        assert fast_match("اطفي الضوء") == "light_off"

    def test_itfee(self):
        assert fast_match("اطفئ الضوء") == "light_off"

    def test_aghliq(self):
        assert fast_match("اغلق الضوء") == "light_off"

    def test_aqfil(self):
        assert fast_match("اقفل الضوء") == "light_off"

    def test_sakker(self):
        assert fast_match("سكر الضو") == "light_off"


# ---------------------------------------------------------------------------
# fast_match — edge cases
# ---------------------------------------------------------------------------

class TestFastMatchEdgeCases:
    def test_empty_string_returns_none(self):
        assert fast_match("") is None

    def test_whitespace_only_returns_none(self):
        assert fast_match("   ") is None

    def test_unrelated_phrase_returns_none(self):
        assert fast_match("open the door") is None

    def test_gibberish_returns_none(self):
        assert fast_match("xyzzy quux blorp") is None

    def test_partial_keyword_does_not_match(self):
        # "شغ" alone should not trigger "شغل"
        assert fast_match("شغ فقط") is None


# ---------------------------------------------------------------------------
# get_command
# ---------------------------------------------------------------------------

class TestGetCommand:
    def test_light_on_returns_correct_cmd(self):
        cmd = get_command("light_on")
        assert cmd is not None
        assert cmd["cmd"] == "LIGHT_ON"

    def test_light_off_returns_correct_cmd(self):
        cmd = get_command("light_off")
        assert cmd is not None
        assert cmd["cmd"] == "LIGHT_OFF"

    def test_unknown_returns_none(self):
        assert get_command("unknown") is None

    def test_invalid_string_returns_none(self):
        assert get_command("start_conveyor") is None


# ---------------------------------------------------------------------------
# Registry structure
# ---------------------------------------------------------------------------

class TestRegistryStructure:
    def test_all_stored_values_have_cmd_key(self):
        for intent, payload in STORED_VALUES.items():
            assert "cmd" in payload, f"Intent '{intent}' is missing 'cmd' key"

    def test_allowed_intents_contains_unknown(self):
        assert "unknown" in ALLOWED_INTENTS

    def test_all_stored_values_in_allowed_intents(self):
        for intent in STORED_VALUES:
            assert intent in ALLOWED_INTENTS

    def test_exactly_two_commands(self):
        assert set(STORED_VALUES.keys()) == {"light_on", "light_off"}
