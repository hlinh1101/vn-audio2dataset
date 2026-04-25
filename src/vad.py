"""Voice activity detection using Silero VAD."""

from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.utils import save_json


logger = logging.getLogger("vn-audio2dataset.vad")


class VadError(RuntimeError):
    """Raised when voice activity detection cannot complete."""


def load_vad_model() -> Any:
    """Load and return the Silero VAD model."""

    try:
        from silero_vad import load_silero_vad
    except ImportError as exc:
        raise VadError(
            "silero-vad is not installed. Run 'pip install -r requirements.txt' "
            "and try again."
        ) from exc

    try:
        return load_silero_vad()
    except Exception as exc:
        raise VadError(f"Failed to load Silero VAD model: {exc}") from exc


def run_vad(audio_path: str | Path, config: AppConfig) -> list[dict[str, float]]:
    """Detect speech spans in a mono 16 kHz WAV file.

    Args:
        audio_path: Path to preprocessed audio_16k.wav.
        config: Application config containing VAD parameters.

    Returns:
        Speech spans with start, end, and duration in seconds.
    """

    wav_path = _validate_vad_audio(audio_path, config.vad.sampling_rate)
    logger.info("Loading Silero VAD model")
    model = load_vad_model()

    try:
        from silero_vad import get_speech_timestamps
    except ImportError as exc:
        raise VadError(
            "silero-vad is not installed. Run 'pip install -r requirements.txt' "
            "and try again."
        ) from exc

    logger.info("Reading VAD audio: %s", wav_path)
    try:
        wav = _read_pcm16_mono_wav(wav_path)
        raw_segments = get_speech_timestamps(
            wav,
            model,
            sampling_rate=config.vad.sampling_rate,
            threshold=config.vad.threshold,
            min_silence_duration_ms=config.vad.min_silence_ms,
            speech_pad_ms=config.vad.speech_pad_ms,
            min_speech_duration_ms=config.vad.min_speech_ms,
            return_seconds=True,
        )
    except Exception as exc:
        raise VadError(f"Silero VAD failed for {wav_path}: {exc}") from exc

    segments = [_normalize_segment(segment) for segment in raw_segments]
    logger.info("Detected %d speech spans", len(segments))
    return segments


def save_vad_segments(
    segments: list[dict[str, float]],
    output_path: str | Path,
) -> Path:
    """Save VAD speech spans as JSON."""

    saved_path = save_json(segments, output_path)
    logger.info("Saved VAD segments: %s", saved_path)
    return saved_path


def _validate_vad_audio(audio_path: str | Path, expected_sample_rate: int) -> Path:
    wav_path = Path(audio_path)
    if not wav_path.exists():
        raise VadError(f"VAD input audio file does not exist: {wav_path}")
    if not wav_path.is_file():
        raise VadError(f"VAD input path is not a file: {wav_path}")
    if wav_path.suffix.lower() != ".wav":
        raise VadError(f"VAD input must be a WAV file: {wav_path}")

    try:
        with wave.open(str(wav_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
    except wave.Error as exc:
        raise VadError(f"VAD input is not a readable PCM WAV file: {wav_path}") from exc

    if sample_width != 2:
        raise VadError(
            f"Unexpected VAD audio sample width for {wav_path}: "
            f"{sample_width} bytes. Expected 16-bit PCM WAV."
        )
    if sample_rate != expected_sample_rate:
        raise VadError(
            f"Unexpected VAD audio sample rate for {wav_path}: "
            f"{sample_rate} Hz. Expected {expected_sample_rate} Hz."
        )
    if channels != 1:
        raise VadError(
            f"Unexpected VAD audio channel count for {wav_path}: "
            f"{channels}. Expected mono audio."
        )

    return wav_path


def _read_pcm16_mono_wav(wav_path: Path) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise VadError(
            "PyTorch is required by silero-vad but is not installed. "
            "Run 'pip install -r requirements.txt' and try again."
        ) from exc

    with wave.open(str(wav_path), "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())

    audio = torch.frombuffer(bytearray(frames), dtype=torch.int16).clone().float()
    return audio / 32768.0


def _normalize_segment(segment: dict[str, Any]) -> dict[str, float]:
    start = float(segment["start"])
    end = float(segment["end"])
    duration = max(0.0, end - start)
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(duration, 3),
    }
