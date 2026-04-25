"""Configuration loading for vn-audio2dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class AudioConfig:
    """Audio processing defaults shared by later pipeline stages."""

    vad_sample_rate: int = 16000
    master_sample_rate: int = 44100
    target_sample_rate: int = 22050
    channels: int = 1
    normalize: bool = True


@dataclass(slots=True)
class SegmentConfig:
    """Natural speech segment boundaries for future segmentation logic."""

    min_sec: float = 3.0
    max_sec: float = 10.0
    ideal_min: float = 4.5
    ideal_max: float = 8.0
    merge_gap_sec: float = 0.5
    force_split_target_sec: float = 8.0

    @property
    def min_seconds(self) -> float:
        return self.min_sec

    @property
    def max_seconds(self) -> float:
        return self.max_sec

    @property
    def max_silence_gap_seconds(self) -> float:
        return self.merge_gap_sec


@dataclass(slots=True)
class VadConfig:
    """Silero VAD parameters."""

    sampling_rate: int = 16000
    min_silence_ms: int = 250
    speech_pad_ms: int = 100
    min_speech_ms: int = 250
    threshold: float = 0.5


@dataclass(slots=True)
class TranscriptionConfig:
    """faster-whisper transcription parameters."""

    model_size: str = "small"
    language: str = "vi"
    beam_size: int = 5
    compute_type: str = "float16"
    device: str = "auto"


@dataclass(slots=True)
class CleaningConfig:
    """Transcript text normalization options."""

    lowercase_text: bool = False
    remove_emojis: bool = True
    strip_quotes: bool = False


@dataclass(slots=True)
class FilterConfig:
    """Rule-based filtering thresholds for cleaned transcript records."""

    min_sec: float = 3.0
    max_sec: float = 10.0
    min_words: int = 3
    max_chars: int = 180
    min_rms: float | None = None
    max_no_speech_prob: float | None = None
    min_avg_logprob: float | None = None
    max_unusual_symbol_ratio: float | None = 0.2


@dataclass(slots=True)
class PathConfig:
    """Project data directories."""

    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    output_dir: Path = Path("data/output")
    rejects_dir: Path = Path("data/rejects")


@dataclass(slots=True)
class LoggingConfig:
    """Logging controls for CLI and pipeline modules."""

    level: str = "INFO"
    log_file: Path | None = None


@dataclass(slots=True)
class AppConfig:
    """Top-level application configuration."""

    project_name: str = "vn-audio2dataset"
    audio: AudioConfig = field(default_factory=AudioConfig)
    segments: SegmentConfig = field(default_factory=SegmentConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _as_dict(value: Any, section_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{section_name}' must be a mapping.")
    return value


def _optional_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value)


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load YAML configuration into structured dataclasses.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        Fully populated application configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the YAML root or known sections have invalid shapes.
    """

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping.")

    audio_raw = _as_dict(raw.get("audio"), "audio")
    segments_raw = _as_dict(raw.get("segments"), "segments")
    vad_raw = _as_dict(raw.get("vad"), "vad")
    transcription_raw = _as_dict(raw.get("transcription"), "transcription")
    cleaning_raw = _as_dict(raw.get("cleaning"), "cleaning")
    filter_raw = _as_dict(raw.get("filter"), "filter")
    paths_raw = _as_dict(raw.get("paths"), "paths")
    logging_raw = _as_dict(raw.get("logging"), "logging")

    return AppConfig(
        project_name=str(raw.get("project_name", "vn-audio2dataset")),
        audio=AudioConfig(
            vad_sample_rate=int(audio_raw.get("vad_sample_rate", 16000)),
            master_sample_rate=int(audio_raw.get("master_sample_rate", 44100)),
            target_sample_rate=int(audio_raw.get("target_sample_rate", 22050)),
            channels=int(audio_raw.get("channels", 1)),
            normalize=bool(audio_raw.get("normalize", True)),
        ),
        segments=SegmentConfig(
            min_sec=float(segments_raw.get("min_sec", segments_raw.get("min_seconds", 3.0))),
            max_sec=float(segments_raw.get("max_sec", segments_raw.get("max_seconds", 10.0))),
            ideal_min=float(segments_raw.get("ideal_min", 4.5)),
            ideal_max=float(segments_raw.get("ideal_max", 8.0)),
            merge_gap_sec=float(
                segments_raw.get(
                    "merge_gap_sec",
                    segments_raw.get("max_silence_gap_seconds", 0.5),
                )
            ),
            force_split_target_sec=float(
                segments_raw.get("force_split_target_sec", 8.0)
            ),
        ),
        vad=VadConfig(
            sampling_rate=int(vad_raw.get("sampling_rate", 16000)),
            min_silence_ms=int(vad_raw.get("min_silence_ms", 250)),
            speech_pad_ms=int(vad_raw.get("speech_pad_ms", 100)),
            min_speech_ms=int(vad_raw.get("min_speech_ms", 250)),
            threshold=float(vad_raw.get("threshold", 0.5)),
        ),
        transcription=TranscriptionConfig(
            model_size=str(transcription_raw.get("model_size", "small")),
            language=str(transcription_raw.get("language", "vi")),
            beam_size=int(transcription_raw.get("beam_size", 5)),
            compute_type=str(transcription_raw.get("compute_type", "float16")),
            device=str(transcription_raw.get("device", "auto")),
        ),
        cleaning=CleaningConfig(
            lowercase_text=bool(cleaning_raw.get("lowercase_text", False)),
            remove_emojis=bool(cleaning_raw.get("remove_emojis", True)),
            strip_quotes=bool(cleaning_raw.get("strip_quotes", False)),
        ),
        filter=FilterConfig(
            min_sec=float(filter_raw.get("min_sec", 3.0)),
            max_sec=float(filter_raw.get("max_sec", 10.0)),
            min_words=int(filter_raw.get("min_words", 3)),
            max_chars=int(filter_raw.get("max_chars", 180)),
            min_rms=_optional_float(filter_raw.get("min_rms")),
            max_no_speech_prob=_optional_float(filter_raw.get("max_no_speech_prob")),
            min_avg_logprob=_optional_float(filter_raw.get("min_avg_logprob")),
            max_unusual_symbol_ratio=_optional_float(
                filter_raw.get("max_unusual_symbol_ratio", 0.2)
            ),
        ),
        paths=PathConfig(
            raw_dir=Path(paths_raw.get("raw_dir", "data/raw")),
            processed_dir=Path(paths_raw.get("processed_dir", "data/processed")),
            output_dir=Path(paths_raw.get("output_dir", "data/output")),
            rejects_dir=Path(paths_raw.get("rejects_dir", "data/rejects")),
        ),
        logging=LoggingConfig(
            level=str(logging_raw.get("level", "INFO")),
            log_file=_optional_path(logging_raw.get("log_file")),
        ),
    )


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
