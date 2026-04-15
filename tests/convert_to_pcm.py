"""
Audio-to-PCM converter for pipeline testing.

Converts every audio file found in SAMPLES_DIR into a raw int16 PCM file
(16 000 Hz, mono) and writes it to OUTPUT_DIR.

Conversion is done via ffmpeg, so all common formats are supported:
WAV, OGG, MP3, FLAC, M4A, OPUS, etc.

Output filename convention:
    <original_stem>.pcm

Usage:
    uv run python tests/convert_to_pcm.py              # default paths
    uv run python tests/convert_to_pcm.py --src path/to/samples --dst path/to/pcm
    uv run python tests/convert_to_pcm.py --rate 16000  # override sample rate

After conversion, test the pipeline on every generated PCM:
    for f in voice_samples/pcm/*.pcm; do
        uv run python main.py --pcm "$f" --language ar
    done
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_REPO_ROOT   = Path(__file__).resolve().parent.parent
SAMPLES_DIR  = _REPO_ROOT / "voice_samples"
OUTPUT_DIR   = _REPO_ROOT / "voice_samples" / "pcm"

# Target format expected by WhisperSTT
TARGET_RATE     = 16_000   # Hz
TARGET_CHANNELS = 1        # mono
TARGET_FORMAT   = "s16le"  # signed 16-bit little-endian PCM

# Audio extensions to scan for (case-insensitive)
AUDIO_EXTENSIONS = {".wav", ".ogg", ".mp3", ".flac", ".m4a", ".opus", ".webm"}


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def convert_to_pcm(
    src: Path,
    dst: Path,
    sample_rate: int = TARGET_RATE,
    channels: int = TARGET_CHANNELS,
) -> None:
    """Convert a single audio file to raw int16 PCM using ffmpeg.

    Resamples to *sample_rate* Hz and downmixes to *channels* channels.
    Overwrites *dst* if it already exists.

    Args:
        src:         Source audio file (any format supported by ffmpeg).
        dst:         Destination .pcm file path.
        sample_rate: Output sample rate in Hz (default 16 000).
        channels:    Output channel count (default 1 = mono).

    Raises:
        RuntimeError: If ffmpeg exits with a non-zero return code.
    """
    cmd = [
        "ffmpeg",
        "-y",                        # overwrite without asking
        "-i", str(src),              # input file
        "-ar", str(sample_rate),     # resample
        "-ac", str(channels),        # channel count
        "-f", TARGET_FORMAT,         # raw PCM output format
        str(dst),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {src.name}:\n"
            + result.stderr.decode(errors="replace")
        )


def collect_audio_files(directory: Path) -> list[Path]:
    """Return all audio files directly inside *directory* (non-recursive).

    Files inside sub-directories (e.g. the pcm/ output folder) are excluded
    so we never re-convert already-converted files.

    Args:
        directory: Folder to scan.

    Returns:
        Sorted list of matching Path objects.
    """
    return sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert audio samples to raw int16 PCM for pipeline testing."
    )
    parser.add_argument(
        "--src", type=Path, default=SAMPLES_DIR,
        help=f"Source directory with audio files (default: {SAMPLES_DIR})",
    )
    parser.add_argument(
        "--dst", type=Path, default=OUTPUT_DIR,
        help=f"Output directory for .pcm files (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--rate", type=int, default=TARGET_RATE,
        help=f"Target sample rate in Hz (default: {TARGET_RATE})",
    )
    args = parser.parse_args()

    # Validate ffmpeg
    if not shutil.which("ffmpeg"):
        print("[ERROR] ffmpeg not found. Install it with: sudo apt install ffmpeg")
        sys.exit(1)

    # Validate source
    if not args.src.is_dir():
        print(f"[ERROR] Source directory not found: {args.src}")
        sys.exit(1)

    # Create output directory
    args.dst.mkdir(parents=True, exist_ok=True)

    # Collect files
    audio_files = collect_audio_files(args.src)
    if not audio_files:
        print(f"[INFO] No audio files found in {args.src}")
        sys.exit(0)

    print(f"[INFO] Found {len(audio_files)} audio file(s) in {args.src}")
    print(f"[INFO] Output → {args.dst}  ({args.rate} Hz, mono, int16 PCM)\n")

    # Convert
    ok = failed = 0
    for src_file in audio_files:
        dst_file = args.dst / (src_file.stem + ".pcm")
        try:
            convert_to_pcm(src_file, dst_file, sample_rate=args.rate)
            size_kb = dst_file.stat().st_size / 1024
            print(f"  [OK]   {src_file.name:30s} → {dst_file.name}  ({size_kb:.1f} KB)")
            ok += 1
        except RuntimeError as exc:
            print(f"  [FAIL] {src_file.name:30s} → {exc}")
            failed += 1

    print(f"\n[DONE] {ok} converted, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
