"""
Intent router module.

Responsibilities:
- Define the two supported intents and their command payloads.
- Provide fast keyword-based intent matching (no LLM required).
- Map an intent string to its command payload.

Typical usage:
    intent = fast_match("شغل الضوء")   # → "light_on"
    cmd    = get_command(intent)         # → {"cmd": "LIGHT_ON"}
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Command registry
# ---------------------------------------------------------------------------

STORED_VALUES: Dict[str, Dict[str, Any]] = {
    "light_on":  {"cmd": "LIGHT_ON"},
    "light_off": {"cmd": "LIGHT_OFF"},
}

ALLOWED_INTENTS: tuple[str, ...] = (*STORED_VALUES.keys(), "unknown")


# ---------------------------------------------------------------------------
# Keyword sets (Arabic + English)
# ---------------------------------------------------------------------------

_KEYWORDS: Dict[str, tuple[str, ...]] = {
    "light_on": (
        # English
        "turn on", "switch on", "light on", "on the light",
        "lights on", "open the light", "open light",
        # Arabic
        "شغل", "اضوي", "اضئ", "اضاء", "اشعل", "ضوي", "افتح الضو",
        "شغل الضو", "اضئ الضو", "فتح الضو",
    ),
    "light_off": (
        # English
        "turn off", "switch off", "light off", "off the light",
        "lights off", "close the light", "close light",
        # Arabic
        "اطفئ", "اطفي", "اغلق", "اقفل", "سكر", "اطفي الضو",
        "اطفئ الضو", "اغلق الضو", "اقفل الضو", "سكر الضو",
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fast_match(text: str) -> Optional[str]:
    """Attempt to identify intent using keyword lookup — no LLM required.

    Checks the lowercased text against the keyword sets for each intent.
    Returns on the first match found.

    Args:
        text: Transcribed speech (Arabic or English, any case).

    Returns:
        Intent string ("light_on" or "light_off") if a keyword matches,
        or None if no keyword matches.
    """
    lowered = text.lower().strip()

    if not lowered:
        return None

    for intent, keywords in _KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return intent

    return None


def get_command(intent: str) -> Optional[Dict[str, Any]]:
    """Look up the command payload for a given intent.

    Args:
        intent: One of the keys in STORED_VALUES ("light_on" or "light_off").

    Returns:
        A dict with at least a "cmd" key, or None if intent is unknown.
    """
    return STORED_VALUES.get(intent)
