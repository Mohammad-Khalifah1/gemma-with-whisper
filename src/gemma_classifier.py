"""
Gemma LLM classifier module.

Responsibilities:
- Load Gemma 4 (2B) locally (CPU, float32).
- Classify a transcript into one of the allowed intents.
- Return structured output as IntentResult.

Typical usage:
    clf = GemmaClassifier(model_path="./models/gemma4_2b")
    clf.load()                           # optional pre-warm
    result = clf.classify("شغل الضوء")
    print(result.intent, result.confidence)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_GEMMA_MODEL = "./models/gemma4_2b"

ALLOWED_INTENTS: tuple[str, ...] = ("light_on", "light_off", "unknown")

_SYSTEM_PROMPT = f"""
You are an intent extractor for a smart-home voice assistant.
Return ONLY valid JSON with exactly these keys:
{{"intent":"<one of: {', '.join(ALLOWED_INTENTS)}>","confidence":0.0}}

Rules:
- "light_on"  → user wants to turn the light ON  (e.g. "turn on the light", "شغل الضوء").
- "light_off" → user wants to turn the light OFF (e.g. "turn off the light", "اطفئ الضوء").
- "unknown"   → anything else.
- Do not add markdown, explanation, or extra text.
""".strip()


@dataclass
class IntentResult:
    """Output of a Gemma classification call."""

    intent: str
    confidence: float
    raw: Dict[str, Any] = field(default_factory=dict)


class GemmaClassifier:
    """Offline LLM intent classifier backed by Gemma 4 (2B)."""

    def __init__(self, model_path: str = DEFAULT_GEMMA_MODEL) -> None:
        self.model_path = model_path
        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Pre-warm the model and tokenizer.

        Call once before the first classify() to avoid a cold-start delay.
        Safe to call multiple times — loads only on the first call.
        """
        self._get_tokenizer_and_model()

    def classify(self, transcript: str) -> IntentResult:
        """Classify a transcript into one of the allowed intents.

        Args:
            transcript: Text produced by STT (Arabic or English).

        Returns:
            IntentResult with .intent, .confidence, and .raw (full JSON dict).

        Raises:
            ValueError: If the model output cannot be parsed as valid JSON.
        """
        tokenizer, model = self._get_tokenizer_and_model()
        prompt = self._build_prompt(tokenizer, transcript)
        decoded = self._generate(tokenizer, model, prompt)
        raw = self._extract_json(decoded)

        return IntentResult(
            intent=str(raw.get("intent", "unknown")).strip(),
            confidence=float(raw.get("confidence", 0.0)),
            raw=raw,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_tokenizer_and_model(self):
        """Lazy-load tokenizer + model (loaded once, reused on subsequent calls)."""
        if self._tokenizer is None or self._model is None:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, local_files_only=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            model.to("cpu")
            model.eval()

            self._tokenizer = tokenizer
            self._model = model

        return self._tokenizer, self._model

    @staticmethod
    def _build_prompt(tokenizer, transcript: str) -> str:
        """Format the chat messages into a model-ready prompt string."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": transcript},
        ]

        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return f"{_SYSTEM_PROMPT}\n\nUser: {transcript}\nAssistant:"

    @staticmethod
    def _generate(tokenizer, model, prompt: str) -> str:
        """Run greedy inference and return only the newly generated tokens.

        Slices off the input tokens from the output so the caller receives
        only what the model produced, not the echoed prompt.
        """
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_len:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """Parse the first valid JSON object found in model output.

        Args:
            text: Raw string from the model decoder.

        Returns:
            Parsed dict from the JSON object.

        Raises:
            ValueError: If no valid JSON object can be found in text.
        """
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        for candidate in reversed(re.findall(r"\{.*?\}", text, flags=re.S)):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        raise ValueError(f"Could not parse JSON from model output:\n{text}")
