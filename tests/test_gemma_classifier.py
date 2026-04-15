"""
Tests for src/gemma_classifier.py

Unit tests mock the model so they run instantly without any hardware.
Integration tests are gated behind a marker and require the real model.

    uv run python -m pytest tests/test_gemma_classifier.py -v               # unit only
    uv run python -m pytest tests/test_gemma_classifier.py -v -m integration # with model
""" 

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.gemma_classifier import GemmaClassifier, IntentResult, ALLOWED_INTENTS


MODEL_PATH = "./models/gemma4_2b"


# ---------------------------------------------------------------------------
# _extract_json — pure Python, no model needed
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_clean_json(self):
        result = GemmaClassifier._extract_json('{"intent":"light_on","confidence":0.95}')
        assert result["intent"] == "light_on"
        assert result["confidence"] == 0.95

    def test_json_embedded_in_prose(self):
        raw = 'Sure! Here is the answer: {"intent":"light_off","confidence":0.8} done.'
        result = GemmaClassifier._extract_json(raw)
        assert result["intent"] == "light_off"

    def test_json_with_surrounding_whitespace(self):
        result = GemmaClassifier._extract_json('  {"intent":"unknown","confidence":0.1}  ')
        assert result["intent"] == "unknown"

    def test_no_json_raises_value_error(self):
        with pytest.raises(ValueError, match="Could not parse JSON"):
            GemmaClassifier._extract_json("This is plain text with no JSON.")

    def test_malformed_json_raises_value_error(self):
        with pytest.raises(ValueError):
            GemmaClassifier._extract_json("{intent: missing_quotes}")


# ---------------------------------------------------------------------------
# _build_prompt — no model inference needed
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_uses_chat_template_when_available(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<formatted_prompt>"
        prompt = GemmaClassifier._build_prompt(mock_tokenizer, "turn on the light")
        mock_tokenizer.apply_chat_template.assert_called_once()
        assert prompt == "<formatted_prompt>"

    def test_falls_back_to_plain_format(self):
        mock_tokenizer = MagicMock(spec=[])    # no apply_chat_template attribute
        prompt = GemmaClassifier._build_prompt(mock_tokenizer, "turn on the light")
        assert "turn on the light" in prompt


# ---------------------------------------------------------------------------
# IntentResult dataclass
# ---------------------------------------------------------------------------

class TestIntentResult:
    def test_all_fields_accessible(self):
        r = IntentResult(intent="light_on", confidence=0.9, raw={"intent": "light_on", "confidence": 0.9})
        assert r.intent == "light_on"
        assert r.confidence == 0.9
        assert r.raw["intent"] == "light_on"

    def test_raw_defaults_to_empty_dict(self):
        r = IntentResult(intent="unknown", confidence=0.0)
        assert r.raw == {}


# ---------------------------------------------------------------------------
# classify() — mocked model
# ---------------------------------------------------------------------------

def _make_clf_with_output(json_output: str) -> GemmaClassifier:
    """Return a GemmaClassifier whose _generate() is patched to return json_output."""
    clf = GemmaClassifier(model_path=MODEL_PATH)
    clf._tokenizer = MagicMock()
    clf._tokenizer.apply_chat_template.return_value = "<prompt>"
    clf._model = MagicMock()
    return clf


class TestClassifyWithMock:
    def test_light_on_intent(self):
        clf = _make_clf_with_output('{"intent":"light_on","confidence":0.97}')
        with patch.object(GemmaClassifier, "_generate",
                          return_value='{"intent":"light_on","confidence":0.97}'):
            result = clf.classify("turn on the light")
        assert isinstance(result, IntentResult)
        assert result.intent == "light_on"
        assert result.confidence == 0.97

    def test_light_off_intent(self):
        clf = _make_clf_with_output('{"intent":"light_off","confidence":0.92}')
        with patch.object(GemmaClassifier, "_generate",
                          return_value='{"intent":"light_off","confidence":0.92}'):
            result = clf.classify("اطفئ الضوء")
        assert result.intent == "light_off"

    def test_unknown_intent(self):
        clf = _make_clf_with_output('{"intent":"unknown","confidence":0.1}')
        with patch.object(GemmaClassifier, "_generate",
                          return_value='{"intent":"unknown","confidence":0.1}'):
            result = clf.classify("blah blah blah")
        assert result.intent == "unknown"

    def test_all_allowed_intents_are_accepted(self):
        for intent in ALLOWED_INTENTS:
            clf = _make_clf_with_output("")
            payload = f'{{"intent":"{intent}","confidence":0.8}}'
            with patch.object(GemmaClassifier, "_generate", return_value=payload):
                result = clf.classify("some text")
            assert result.intent == intent


# ---------------------------------------------------------------------------
# ALLOWED_INTENTS structure
# ---------------------------------------------------------------------------

class TestAllowedIntents:
    def test_contains_light_on(self):
        assert "light_on" in ALLOWED_INTENTS

    def test_contains_light_off(self):
        assert "light_off" in ALLOWED_INTENTS

    def test_contains_unknown(self):
        assert "unknown" in ALLOWED_INTENTS

    def test_exactly_three_intents(self):
        assert len(ALLOWED_INTENTS) == 3


# ---------------------------------------------------------------------------
# Integration test — requires real model at ./models/gemma4_2b
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestClassifyIntegration:
    def test_arabic_light_on(self):
        from pathlib import Path
        if not Path(MODEL_PATH).exists():
            pytest.skip(f"Gemma model not found at {MODEL_PATH}")

        clf = GemmaClassifier(model_path=MODEL_PATH)
        result = clf.classify("شغل الضوء")
        assert isinstance(result, IntentResult)
        assert result.intent in ALLOWED_INTENTS
        assert 0.0 <= result.confidence <= 1.0

    def test_arabic_light_off(self):
        from pathlib import Path
        if not Path(MODEL_PATH).exists():
            pytest.skip(f"Gemma model not found at {MODEL_PATH}")

        clf = GemmaClassifier(model_path=MODEL_PATH)
        result = clf.classify("اطفئ الضوء")
        assert isinstance(result, IntentResult)
        assert result.intent in ALLOWED_INTENTS
