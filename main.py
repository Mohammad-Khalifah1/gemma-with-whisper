from __future__ import annotations

import argparse
import json
import re
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_WHISPER_MODEL = "./models/whisper"
DEFAULT_GEMMA_MODEL = "./models/gemma4_2b"

STORED_VALUES: Dict[str, Dict[str, Any]] = {
    "start_conveyor": {"cmd": "CONVEYOR_START", "speed": 30},
    "stop_conveyor": {"cmd": "CONVEYOR_STOP", "speed": 0},
    "open_gripper": {"cmd": "GRIPPER_OPEN", "value": 1},
    "close_gripper": {"cmd": "GRIPPER_CLOSE", "value": 0},
    "status": {"cmd": "STATUS_QUERY"},
}

ALLOWED_INTENTS = ", ".join(STORED_VALUES.keys()) + ", unknown"

SYSTEM_PROMPT = f"""
You are an intent extractor for a voice assistant.
Return ONLY valid JSON with exactly these keys:
{{"intent":"<one of: {ALLOWED_INTENTS}>","confidence":0.0}}

Rules:
- Choose the closest matching intent from the allowed list.
- If nothing matches, use "unknown".
- Do not add markdown, explanation, or extra text.
""".strip()


def pcm_raw_to_wav(
    pcm_path: str | Path,
    wav_path: str | Path,
    sample_rate: int = 16000,
    channels: int = 1,
) -> None:
    pcm_path = Path(pcm_path)
    wav_path = Path(wav_path)

    raw = pcm_path.read_bytes()
    if len(raw) == 0:
        raise ValueError(f"PCM file is empty: {pcm_path}")

    if len(raw) % 2 != 0:
        raise ValueError("PCM data length is not even. Expected int16 PCM (2 bytes per sample).")

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw)


def transcribe_pcm(
    pcm_path: str | Path,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    sample_rate: int = 16000,
    channels: int = 1,
    language: Optional[str] = None,
) -> Tuple[str, Optional[str], Optional[float]]:
    device = "cpu"
    compute_type = "int8"

    model = WhisperModel(whisper_model, device=device, compute_type=compute_type)

    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "input.wav"
        pcm_raw_to_wav(pcm_path, wav_path, sample_rate=sample_rate, channels=channels)

        segments, info = model.transcribe(
            str(wav_path),
            language=language,
            vad_filter=True,
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()

    return text, getattr(info, "language", None), getattr(info, "language_probability", None)


def fast_match(text: str) -> Optional[str]:
    t = text.lower().strip()

    if not t:
        return None

    if any(k in t for k in ["start", "ابدأ", "شغل", "تشغيل", "ابد"]):
        return "start_conveyor"

    if any(k in t for k in ["stop", "وقف", "اطف", "إيقاف", "اوقف"]):
        return "stop_conveyor"

    if any(k in t for k in ["open", "افتح", "فك", "فتح"]):
        return "open_gripper"

    if any(k in t for k in ["close", "سكر", "اغلق", "إغلاق", "اقفل"]):
        return "close_gripper"

    if any(k in t for k in ["status", "حالة", "شو الحالة", "ما الحالة"]):
        return "status"

    return None


def load_gemma(model_id: str = DEFAULT_GEMMA_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    model.to("cpu")
    model.eval()
    return tokenizer, model


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    matches = re.findall(r"\{.*?\}", text, flags=re.S)
    for candidate in reversed(matches):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not parse JSON from model output:\n{text}")


def classify_with_gemma(
    transcript: str,
    tokenizer,
    model,
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = f"{SYSTEM_PROMPT}\n\nUser: {transcript}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return extract_json_object(decoded)


def main() -> None:
    parser = argparse.ArgumentParser(description="PCM -> STT -> fast match -> Gemma intent extraction")
    parser.add_argument("--pcm", required=True, help="Path to raw PCM file")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--language", default=None, help="Optional: ar or en")
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL, help="Local Whisper model path")
    parser.add_argument("--gemma-model", default=DEFAULT_GEMMA_MODEL, help="Local Gemma model path")
    args = parser.parse_args()

    transcript, detected_lang, lang_prob = transcribe_pcm(
        pcm_path=args.pcm,
        whisper_model=args.whisper_model,
        sample_rate=args.sample_rate,
        channels=args.channels,
        language=args.language,
    )

    print(f"[STT] language={detected_lang} prob={lang_prob}")
    print(f"[STT] text={transcript}")

    if not transcript.strip():
        print("[MATCH] unknown")
        return

    intent = fast_match(transcript)
    result: Dict[str, Any]

    if intent is not None:
        result = {"intent": intent, "confidence": 1.0}
        print("[FAST] matched without Gemma")
    else:
        tokenizer, model = load_gemma(args.gemma_model)
        result = classify_with_gemma(transcript, tokenizer, model)
        intent = str(result.get("intent", "unknown")).strip()

    matched = STORED_VALUES.get(intent)

    print(f"[GEMMA] intent={intent}")
    print(f"[GEMMA] raw={result}")

    if matched is None:
        print("[MATCH] unknown")
    else:
        print("[MATCH]", json.dumps(matched, ensure_ascii=False))


if __name__ == "__main__":
    main()