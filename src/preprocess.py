"""Audio preprocessing stage.

This module prepares two canonical WAV files from a source audio file:

- audio_16k.wav: mono 16 kHz PCM for VAD and ASR
- audio_master.wav: high-quality PCM WAV for later dataset cutting
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.config import AppConfig
from src.utils import ensure_dir, safe_stem


SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac"}


class PreprocessError(RuntimeError):
    """Raised when audio preprocessing cannot complete."""


@dataclass(frozen=True, slots=True)
class PreprocessOutputs:
    """Paths produced by preprocessing."""

    audio_16k: Path
    audio_master: Path


def validate_input_audio(path: str | Path) -> Path:
    """Validate the input audio path and return it as a Path."""

    audio_path = Path(path)
    if not audio_path.exists():
        raise PreprocessError(f"Input audio file does not exist: {audio_path}")
    if not audio_path.is_file():
        raise PreprocessError(f"Input audio path is not a file: {audio_path}")
    if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise PreprocessError(
            f"Unsupported audio format '{audio_path.suffix}'. "
            f"Supported formats: {supported}"
        )
    return audio_path


def run_ffmpeg_command(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run an FFmpeg command and raise a helpful error on failure."""

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise PreprocessError(
            "FFmpeg was not found. Install FFmpeg and ensure 'ffmpeg' is "
            "available on PATH, then run this command again."
        ) from exc

    if result.returncode != 0:
        stderr = result.stderr.strip() or "No FFmpeg error output was captured."
        raise PreprocessError(f"FFmpeg command failed: {' '.join(cmd)}\n{stderr}")

    return result


def convert_to_16k_mono(input_path: str | Path, output_path: str | Path) -> Path:
    """Convert source audio to mono 16 kHz PCM WAV for VAD and ASR."""

    source = Path(input_path)
    target = Path(output_path)
    ensure_dir(target.parent)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(target),
    ]
    run_ffmpeg_command(cmd)
    return target


def convert_to_master_wav(
    input_path: str | Path,
    output_path: str | Path,
    sample_rate: int | None = None,
) -> Path:
    """Convert source audio to a high-quality PCM WAV master file."""

    source = Path(input_path)
    target = Path(output_path)
    ensure_dir(target.parent)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
    ]
    if sample_rate is not None:
        cmd.extend(["-ar", str(sample_rate)])
    cmd.extend(["-c:a", "pcm_s16le", str(target)])

    run_ffmpeg_command(cmd)
    return target


def preprocess_audio(
    input_path: str | Path,
    working_dir: str | Path,
    config: AppConfig,
    logger: logging.Logger | None = None,
) -> PreprocessOutputs:
    """Preprocess input audio into canonical intermediate WAV files."""

    source = validate_input_audio(input_path)
    audio_working_dir = ensure_dir(Path(working_dir) / safe_stem(source))
    audio_16k_path = audio_working_dir / "audio_16k.wav"
    audio_master_path = audio_working_dir / "audio_master.wav"

    if logger is not None:
        logger.info("Preprocessing audio: %s", source)
        logger.info("Working directory: %s", audio_working_dir)

    convert_to_16k_mono(source, audio_16k_path)
    convert_to_master_wav(
        source,
        audio_master_path,
        sample_rate=config.audio.master_sample_rate,
    )

    if logger is not None:
        logger.info("Created VAD/ASR audio: %s", audio_16k_path)
        logger.info("Created master audio: %s", audio_master_path)

    return PreprocessOutputs(
        audio_16k=audio_16k_path,
        audio_master=audio_master_path,
    )
