"""
Tests for src/whisper_stt.py

These tests cover the pure-Python logic (PCM → WAV conversion, validation)
without loading the Whisper model, so they run fast and offline.

The integration test that calls .transcribe() is gated behind a marker so it
only runs when the actual model files are present:

    uv run pytest tests/test_whisper_stt.py -v                     # unit only
    uv run pytest tests/test_whisper_stt.py -v -m integration      # with model
"""

from __future__ import annotations

import struct
import tempfile
import wave
from pathlib import Path

import pytest

from src.whisper_stt import WhisperSTT, TranscriptResult


MODEL_PATH = "./models/whisper"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pcm(tmp_path: Path, n_samples: int = 100) -> Path:
    """Write a minimal valid int16 PCM file."""
    pcm_path = tmp_path / "test.pcm"
    pcm_path.write_bytes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))
    return pcm_path


def _read_wav_header(wav_path: Path):
    """Return (nchannels, sampwidth, framerate, nframes) from a WAV file."""
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()


# ---------------------------------------------------------------------------
# _pcm_to_wav — core conversion logic
# ---------------------------------------------------------------------------

class TestPcmToWav:
    def test_valid_pcm_creates_wav(self, tmp_path):
        pcm = _make_pcm(tmp_path)
        wav = tmp_path / "out.wav"
        WhisperSTT._pcm_to_wav(pcm, wav, sample_rate=16000, channels=1)
        assert wav.exists()

    def test_wav_header_values(self, tmp_path):
        pcm = _make_pcm(tmp_path, n_samples=200)
        wav = tmp_path / "out.wav"
        WhisperSTT._pcm_to_wav(pcm, wav, sample_rate=16000, channels=1)
        nch, sampwidth, framerate, nframes = _read_wav_header(wav)
        assert nch == 1
        assert sampwidth == 2          # int16
        assert framerate == 16000
        assert nframes == 200

    def test_custom_sample_rate(self, tmp_path):
        pcm = _make_pcm(tmp_path)
        wav = tmp_path / "out.wav"
        WhisperSTT._pcm_to_wav(pcm, wav, sample_rate=44100, channels=1)
        _, _, framerate, _ = _read_wav_header(wav)
        assert framerate == 44100

    def test_empty_pcm_raises_value_error(self, tmp_path):
        pcm = tmp_path / "empty.pcm"
        pcm.write_bytes(b"")
        wav = tmp_path / "out.wav"
        with pytest.raises(ValueError, match="empty"):
            WhisperSTT._pcm_to_wav(pcm, wav, sample_rate=16000, channels=1)

    def test_odd_byte_count_raises_value_error(self, tmp_path):
        pcm = tmp_path / "odd.pcm"
        pcm.write_bytes(b"\x00\x01\x02")   # 3 bytes → odd
        wav = tmp_path / "out.wav"
        with pytest.raises(ValueError, match="not even"):
            WhisperSTT._pcm_to_wav(pcm, wav, sample_rate=16000, channels=1)


# ---------------------------------------------------------------------------
# transcribe() — input validation (no model needed)
# ---------------------------------------------------------------------------

class TestTranscribeValidation:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        stt = WhisperSTT(model_path=MODEL_PATH)
        with pytest.raises(FileNotFoundError):
            stt.transcribe(tmp_path / "nonexistent.pcm")

    def test_empty_pcm_raises_value_error(self, tmp_path):
        pcm = tmp_path / "empty.pcm"
        pcm.write_bytes(b"")
        stt = WhisperSTT(model_path=MODEL_PATH)
        with pytest.raises(ValueError, match="empty"):
            stt.transcribe(pcm)


# ---------------------------------------------------------------------------
# TranscriptResult dataclass
# ---------------------------------------------------------------------------

class TestTranscriptResult:
    def test_fields_accessible(self):
        r = TranscriptResult(text="hello", language="en", language_probability=0.99)
        assert r.text == "hello"
        assert r.language == "en"
        assert r.language_probability == 0.99

    def test_optional_fields_can_be_none(self):
        r = TranscriptResult(text="", language=None, language_probability=None)
        assert r.language is None
        assert r.language_probability is None


# ---------------------------------------------------------------------------
# _deduplicate — pure Python, no model needed
# ---------------------------------------------------------------------------

class TestDeduplicate:
    def test_removes_consecutive_duplicates(self):
        assert WhisperSTT._deduplicate("اطفئ الضوء الضوء الضوء") == "اطفئ الضوء"

    def test_removes_english_duplicates(self):
        assert WhisperSTT._deduplicate("turn on the the light") == "turn on the light"

    def test_no_duplicates_unchanged(self):
        assert WhisperSTT._deduplicate("شغل الضوء") == "شغل الضوء"

    def test_single_word_unchanged(self):
        assert WhisperSTT._deduplicate("hello") == "hello"

    def test_empty_string_unchanged(self):
        assert WhisperSTT._deduplicate("") == ""

    def test_non_consecutive_duplicates_kept(self):
        # "الضوء" appears twice but not consecutively — both must stay
        assert WhisperSTT._deduplicate("شغل الضوء واطفئ الضوء") == "شغل الضوء واطفئ الضوء"


# ---------------------------------------------------------------------------
# Integration tests — require real Whisper model + voice_samples/pcm/
# ---------------------------------------------------------------------------

# Expected transcripts for each voice sample.
# key   = filename stem inside voice_samples/pcm/
# value = (expected_text, expected_intent)
_EXPECTED: dict[str, tuple[str, str]] = {
    "speech":    ("turn the lights on", "light_on"),
    "speech(1)": ("شغل الضوء",          "light_on"),
    "speech(2)": ("اضوي الضوء",         "light_on"),
    "speech(3)": ("اطفئ الضوء",         "light_off"),
}


def _skip_if_missing(model_path: str, pcm_path: Path) -> None:
    """Skip the test if the model or PCM file is not present."""
    if not Path(model_path).exists():
        pytest.skip(f"Whisper model not found at {model_path}")
    if not pcm_path.exists():
        pytest.skip(f"PCM file not found: {pcm_path}")


@pytest.mark.integration
class TestTranscribeIntegration:
    """Run the real Whisper model against every voice sample."""

    @pytest.fixture(scope="class")
    def stt(self):
        """Shared WhisperSTT instance — model loaded once for the whole class."""
        if not Path(MODEL_PATH).exists():
            pytest.skip(f"Whisper model not found at {MODEL_PATH}")
        return WhisperSTT(model_path=MODEL_PATH)

    @pytest.mark.parametrize("stem,expected", [
        pytest.param(stem, expected, id=stem)
        for stem, expected in _EXPECTED.items()
    ])
    def test_transcript_and_intent(self, stt, stem, expected):
        """Transcribe a sample and verify text + intent match expectations."""
        from src.intent_router import fast_match

        expected_text, expected_intent = expected
        pcm_path = Path("voice_samples/pcm") / f"{stem}.pcm"
        _skip_if_missing(MODEL_PATH, pcm_path)

        result = stt.transcribe(pcm_path)

        assert isinstance(result, TranscriptResult)
        assert result.text == expected_text, (
            f"[{stem}] transcript mismatch:\n"
            f"  got:      {result.text!r}\n"
            f"  expected: {expected_text!r}"
        )

        intent = fast_match(result.text)
        assert intent == expected_intent, (
            f"[{stem}] intent mismatch:\n"
            f"  got:      {intent!r}\n"
            f"  expected: {expected_intent!r}"
        )
