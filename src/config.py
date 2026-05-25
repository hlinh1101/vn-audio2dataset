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
    max_sec: float = 15.0
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
    """Transcription parameters."""

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
class AudioQualityConfig:
    """Rule-based thresholds for pre-transcription audio quality gating."""

    min_sec: float = 3.0
    max_sec: float = 10.0
    silence_threshold_dbfs: float = -55.0
    min_rms_dbfs: float = -28.0
    review_min_rms_dbfs: float = -24.0
    max_silence_ratio: float = 0.35
    review_silence_ratio: float = 0.20
    max_leading_silence_sec: float = 0.35
    review_leading_silence_sec: float = 0.15
    max_trailing_silence_sec: float = 0.35
    review_trailing_silence_sec: float = 0.15
    max_clipping_ratio: float = 0.01
    review_clipping_ratio: float = 0.002
    max_spectral_flatness: float = 0.20
    review_spectral_flatness: float = 0.10
    max_high_freq_energy_ratio: float = 0.30
    review_high_freq_energy_ratio: float = 0.18


@dataclass(slots=True)
class SttSegmentationConfig:
    """Transcript-first segmentation controls for ElevenLabs Scribe output."""

    model_id: str = "scribe_v2"
    timestamps_granularity: str = "word"
    diarize: bool = True
    tag_audio_events: bool = True
    num_speakers: int | None = None
    diarization_threshold: float | None = None
    dominant_speaker: str = "auto"
    speaker_selection_mode: str = "auto"
    default_target_speaker: str = "auto"
    require_manual_speaker: bool = False
    per_file_target_speakers: dict[str, str] = field(default_factory=dict)
    min_speaker_share: float = 0.45
    boundary_pad_sec: float = 0.05
    boundary_guard_sec: float = 0.20
    max_word_gap_sec: float = 0.80
    preferred_min_sec: float = 5.0
    preferred_max_sec: float = 10.0
    semantic_max_sec: float = 15.0
    sentence_punctuation_weight: float = 8.0
    clause_punctuation_weight: float = 4.0
    pause_strong_sec: float = 0.45
    pause_medium_sec: float = 0.25
    pause_strong_weight: float = 5.0
    pause_medium_weight: float = 2.5
    min_boundary_score: float = 7.0
    allow_short_clips: bool = False
    protect_named_entities: bool = True
    protect_connector_phrases: bool = True
    min_words: int = 3
    min_avg_logprob: float | None = None


@dataclass(slots=True)
class FilterConfig:
    """Rule-based filtering thresholds for cleaned transcript records."""

    min_sec: float = 3.0
    max_sec: float = 15.0
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
    transcription_backend: str = "faster_whisper"
    audio: AudioConfig = field(default_factory=AudioConfig)
    segments: SegmentConfig = field(default_factory=SegmentConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    audio_quality: AudioQualityConfig = field(default_factory=AudioQualityConfig)
    stt_segmentation: SttSegmentationConfig = field(default_factory=SttSegmentationConfig)
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
    audio_quality_raw = _as_dict(raw.get("audio_quality"), "audio_quality")
    stt_segmentation_raw = _as_dict(raw.get("stt_segmentation"), "stt_segmentation")
    filter_raw = _as_dict(raw.get("filter"), "filter")
    paths_raw = _as_dict(raw.get("paths"), "paths")
    logging_raw = _as_dict(raw.get("logging"), "logging")
    transcription_backend = _normalize_transcription_backend(
        raw.get(
            "transcription_backend",
            transcription_raw.get("backend", "faster_whisper"),
        )
    )

    return AppConfig(
        project_name=str(raw.get("project_name", "vn-audio2dataset")),
        transcription_backend=transcription_backend,
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
        audio_quality=AudioQualityConfig(
            min_sec=float(audio_quality_raw.get("min_sec", 3.0)),
            max_sec=float(audio_quality_raw.get("max_sec", 10.0)),
            silence_threshold_dbfs=float(
                audio_quality_raw.get("silence_threshold_dbfs", -55.0)
            ),
            min_rms_dbfs=float(audio_quality_raw.get("min_rms_dbfs", -28.0)),
            review_min_rms_dbfs=float(
                audio_quality_raw.get("review_min_rms_dbfs", -24.0)
            ),
            max_silence_ratio=float(audio_quality_raw.get("max_silence_ratio", 0.35)),
            review_silence_ratio=float(
                audio_quality_raw.get("review_silence_ratio", 0.20)
            ),
            max_leading_silence_sec=float(
                audio_quality_raw.get("max_leading_silence_sec", 0.35)
            ),
            review_leading_silence_sec=float(
                audio_quality_raw.get("review_leading_silence_sec", 0.15)
            ),
            max_trailing_silence_sec=float(
                audio_quality_raw.get("max_trailing_silence_sec", 0.35)
            ),
            review_trailing_silence_sec=float(
                audio_quality_raw.get("review_trailing_silence_sec", 0.15)
            ),
            max_clipping_ratio=float(audio_quality_raw.get("max_clipping_ratio", 0.01)),
            review_clipping_ratio=float(
                audio_quality_raw.get("review_clipping_ratio", 0.002)
            ),
            max_spectral_flatness=float(
                audio_quality_raw.get("max_spectral_flatness", 0.20)
            ),
            review_spectral_flatness=float(
                audio_quality_raw.get("review_spectral_flatness", 0.10)
            ),
            max_high_freq_energy_ratio=float(
                audio_quality_raw.get("max_high_freq_energy_ratio", 0.30)
            ),
            review_high_freq_energy_ratio=float(
                audio_quality_raw.get("review_high_freq_energy_ratio", 0.18)
            ),
        ),
        stt_segmentation=SttSegmentationConfig(
            model_id=str(stt_segmentation_raw.get("model_id", "scribe_v2")),
            timestamps_granularity=str(
                stt_segmentation_raw.get("timestamps_granularity", "word")
            ),
            diarize=bool(stt_segmentation_raw.get("diarize", True)),
            tag_audio_events=bool(stt_segmentation_raw.get("tag_audio_events", True)),
            num_speakers=_optional_int(stt_segmentation_raw.get("num_speakers")),
            diarization_threshold=_optional_float(
                stt_segmentation_raw.get("diarization_threshold")
            ),
            dominant_speaker=str(
                stt_segmentation_raw.get("dominant_speaker", "auto")
            ),
            speaker_selection_mode=str(
                stt_segmentation_raw.get("speaker_selection_mode", "auto")
            ),
            default_target_speaker=str(
                stt_segmentation_raw.get(
                    "default_target_speaker",
                    stt_segmentation_raw.get("dominant_speaker", "auto"),
                )
            ),
            require_manual_speaker=bool(
                stt_segmentation_raw.get("require_manual_speaker", False)
            ),
            per_file_target_speakers=_string_mapping(
                stt_segmentation_raw.get("per_file_target_speakers")
            ),
            min_speaker_share=float(
                stt_segmentation_raw.get("min_speaker_share", 0.45)
            ),
            boundary_pad_sec=float(
                stt_segmentation_raw.get("boundary_pad_sec", 0.05)
            ),
            boundary_guard_sec=float(
                stt_segmentation_raw.get("boundary_guard_sec", 0.20)
            ),
            max_word_gap_sec=float(
                stt_segmentation_raw.get("max_word_gap_sec", 0.80)
            ),
            preferred_min_sec=float(
                stt_segmentation_raw.get("preferred_min_sec", 5.0)
            ),
            preferred_max_sec=float(
                stt_segmentation_raw.get("preferred_max_sec", 10.0)
            ),
            semantic_max_sec=float(
                stt_segmentation_raw.get("semantic_max_sec", 15.0)
            ),
            sentence_punctuation_weight=float(
                stt_segmentation_raw.get("sentence_punctuation_weight", 8.0)
            ),
            clause_punctuation_weight=float(
                stt_segmentation_raw.get("clause_punctuation_weight", 4.0)
            ),
            pause_strong_sec=float(
                stt_segmentation_raw.get("pause_strong_sec", 0.45)
            ),
            pause_medium_sec=float(
                stt_segmentation_raw.get("pause_medium_sec", 0.25)
            ),
            pause_strong_weight=float(
                stt_segmentation_raw.get("pause_strong_weight", 5.0)
            ),
            pause_medium_weight=float(
                stt_segmentation_raw.get("pause_medium_weight", 2.5)
            ),
            min_boundary_score=float(
                stt_segmentation_raw.get("min_boundary_score", 7.0)
            ),
            allow_short_clips=bool(
                stt_segmentation_raw.get("allow_short_clips", False)
            ),
            protect_named_entities=bool(
                stt_segmentation_raw.get("protect_named_entities", True)
            ),
            protect_connector_phrases=bool(
                stt_segmentation_raw.get("protect_connector_phrases", True)
            ),
            min_words=int(stt_segmentation_raw.get("min_words", 3)),
            min_avg_logprob=_optional_float(
                stt_segmentation_raw.get("min_avg_logprob")
            ),
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


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _string_mapping(value: Any) -> dict[str, str]:
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise ValueError("Config value must be a mapping.")
    return {
        str(key): str(item)
        for key, item in value.items()
        if key not in (None, "") and item not in (None, "")
    }


def _normalize_transcription_backend(value: Any) -> str:
    backend = str(value or "faster_whisper").strip().lower().replace("-", "_")
    if backend not in {"faster_whisper", "elevenlabs"}:
        raise ValueError(
            "transcription_backend must be either 'faster_whisper' or 'elevenlabs'."
        )
    return backend
