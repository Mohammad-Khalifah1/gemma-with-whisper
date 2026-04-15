"""
Whisper STT module.

Responsibilities:
- Convert raw PCM audio to WAV.
- Transcribe audio using faster-whisper (CPU, int8).

Accuracy tuning applied:
- language=None by default  → Whisper auto-detects; avoids misreading English
  commands as Arabic when language="ar" is forced.
- initial_prompt             → primes the decoder with expected smart-home
  phrases (both Arabic and English) so domain vocabulary is preferred.
- hotwords                   → boosts key command words during beam search.
- condition_on_previous_text=False → prevents hallucination propagation across
  segments.
- temperature=0.0            → fully deterministic; skips the fallback retry
  ladder that can introduce noise on short clips.
- vad_parameters             → tighter silence thresholds so short voice
  commands are not clipped or padded with silent noise.

Typical usage:
    stt = WhisperSTT(model_path="./models/whisper")
    result = stt.transcribe("audio.pcm")        # auto-detect language
    result = stt.transcribe("audio.pcm", language="ar")  # force Arabic
    print(result.text, result.language)
"""

from __future__ import annotations

import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel


DEFAULT_WHISPER_MODEL = "./models/whisper"

# Initial prompt — primes the Whisper decoder with domain-relevant vocabulary.
# Having these phrases in both Arabic and English helps the model weight smart-
# home command words higher during beam search regardless of the spoken language.
_INITIAL_PROMPT = (
    "turn on the light, turn off the light, switch on, switch off, "
    "lights on, lights off, "
    "شغل الضوء، اطفئ الضوء، اضوي الضوء، اغلق الضوء، "
    "اضئ الضوء، اشعل الضوء، اطفي الضوء، اقفل الضوء"
)

# Hotwords — boosted during beam search scoring.
_HOTWORDS = (
    "شغل اطفئ اطفي اضوي اضئ اشعل اغلق اقفل الضوء الضو "
    "turn switch light lights on off"
)

# VAD parameters — tighter thresholds suited for short voice commands.
_VAD_PARAMETERS = {
    "threshold": 0.3,             # lower → catches quieter speech onsets
    "min_speech_duration_ms": 100,
    "min_silence_duration_ms": 300,
    "speech_pad_ms": 200,         # padding around detected speech
}


@dataclass
class TranscriptResult:
    """Output of a transcription call."""

    text: str
    language: Optional[str]
    language_probability: Optional[float]


class WhisperSTT:
    """Wrapper around faster-whisper for offline CPU transcription."""

    def __init__(
        self,
        model_path: str = DEFAULT_WHISPER_MODEL,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self._model: Optional[WhisperModel] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        pcm_path: str | Path,
        sample_rate: int = 16000,
        channels: int = 1,
        language: Optional[str] = None,
    ) -> TranscriptResult:
        """Transcribe a raw PCM file and return a TranscriptResult.

        Args:
            pcm_path:    Path to the raw PCM file (int16, little-endian).
            sample_rate: PCM sample rate in Hz (default 16 000).
            channels:    Number of audio channels (default 1 = mono).
            language:    BCP-47 language code hint ("ar", "en", …).
                         Defaults to None — Whisper auto-detects the language.
                         Only force a language if you are certain the audio is
                         monolingual; forcing the wrong language degrades accuracy.

        Returns:
            TranscriptResult with .text, .language, .language_probability.

        Raises:
            FileNotFoundError: If pcm_path does not exist.
            ValueError:        If the PCM file is empty or has an odd byte count.
        """
        pcm_path = Path(pcm_path)

        if not pcm_path.exists():
            raise FileNotFoundError(f"PCM file not found: {pcm_path}")

        model = self._get_model()

        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "input.wav"
            self._pcm_to_wav(pcm_path, wav_path, sample_rate, channels)

            segments, info = model.transcribe(
                str(wav_path),
                # Language — auto-detect by default to handle mixed-language commands.
                language=language,
                # Beam search — default beam_size=5 is already good; keep explicit.
                beam_size=5,
                # Deterministic decoding — skip the temperature retry ladder that
                # adds noise on short clips.
                temperature=0.0,
                # Domain priming — steers the decoder toward smart-home vocabulary.
                initial_prompt=_INITIAL_PROMPT,
                # Hotword boosting during beam search.
                hotwords=_HOTWORDS,
                # Segment independence — prevents one bad segment from poisoning the next.
                condition_on_previous_text=False,
                # Hallucination suppression — Whisper tends to repeat tokens in
                # silent regions; this threshold (seconds) triggers internal
                # silence detection and drops those repeated segments.
                hallucination_silence_threshold=0.5,
                # VAD — tighter thresholds for short voice commands.
                vad_filter=True,
                vad_parameters=_VAD_PARAMETERS,
            )

            raw_text = " ".join(seg.text.strip() for seg in segments).strip()
            text = self._deduplicate(raw_text)

        return TranscriptResult(
            text=text,
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model(self) -> WhisperModel:
        """Lazy-load the Whisper model (loaded once, reused on subsequent calls)."""
        if self._model is None:
            self._model = WhisperModel(
                self.model_path,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    @staticmethod
    def _pcm_to_wav(
        pcm_path: Path,
        wav_path: Path,
        sample_rate: int,
        channels: int,
    ) -> None:
        """Write a WAV header around raw int16 PCM data.

        Args:
            pcm_path:    Source PCM file.
            wav_path:    Destination WAV file (will be created/overwritten).
            sample_rate: Sample rate in Hz.
            channels:    Number of audio channels.

        Raises:
            ValueError: If PCM data is empty or has an odd byte count.
        """
        raw = pcm_path.read_bytes()

        if len(raw) == 0:
            raise ValueError(f"PCM file is empty: {pcm_path}")
        if len(raw) % 2 != 0:
            raise ValueError(
                "PCM data length is not even — expected int16 PCM (2 bytes/sample)."
            )

        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)           # int16 → 2 bytes per sample
            wf.setframerate(sample_rate)
            wf.writeframes(raw)

    @staticmethod
    def _deduplicate(text: str) -> str:
        """Remove consecutively repeated words from a transcript.

        Whisper sometimes hallucinates by repeating the last word or phrase
        when it encounters silence at the end of a short recording.
        This method collapses runs of identical consecutive tokens into one.

        Examples:
            "اطفئ الضوء الضوء الضوء" → "اطفئ الضوء"
            "turn on the the light"   → "turn on the light"
            "hello"                   → "hello"

        Args:
            text: Raw transcript string from Whisper.

        Returns:
            Cleaned transcript with consecutive duplicate words removed.
        """
        words = text.split()
        if not words:
            return text

        deduped = [words[0]]
        for word in words[1:]:
            if word != deduped[-1]:
                deduped.append(word)

        return " ".join(deduped)
