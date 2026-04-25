"""Transcribe exported WAV segments with faster-whisper."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.utils import ensure_dir, load_json


logger = logging.getLogger("vn-audio2dataset.transcribe")


class TranscriptionError(RuntimeError):
    """Raised when transcription setup cannot complete."""


def load_whisper_model(config: AppConfig) -> Any:
    """Load a faster-whisper model once for the current pipeline run."""

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError(
            "faster-whisper is not installed. Run 'pip install -r requirements.txt' "
            "and try again."
        ) from exc

    device = _resolve_device(config.transcription.device)
    compute_type = _resolve_compute_type(config.transcription.compute_type, device)

    logger.info(
        "Loading faster-whisper model '%s' on %s with compute_type=%s",
        config.transcription.model_size,
        device,
        compute_type,
    )
    try:
        return WhisperModel(
            config.transcription.model_size,
            device=device,
            compute_type=compute_type,
        )
    except Exception as exc:
        if device == "cuda":
            logger.warning(
                "Failed to load faster-whisper on CUDA, falling back to CPU: %s",
                exc,
            )
            return WhisperModel(
                config.transcription.model_size,
                device="cpu",
                compute_type=_resolve_compute_type(
                    config.transcription.compute_type,
                    "cpu",
                ),
            )
        raise TranscriptionError(f"Failed to load faster-whisper model: {exc}") from exc


def transcribe_segment(audio_path: str | Path, model: Any, config: AppConfig) -> dict[str, Any]:
    """Transcribe one exported WAV segment."""

    path = Path(audio_path)
    if not path.exists():
        raise TranscriptionError(f"Audio segment file does not exist: {path}")
    if not path.is_file():
        raise TranscriptionError(f"Audio segment path is not a file: {path}")

    segments, info = model.transcribe(
        str(path),
        language=config.transcription.language,
        beam_size=config.transcription.beam_size,
    )
    segment_items = list(segments)
    text = " ".join(item.text.strip() for item in segment_items if item.text).strip()

    return {
        "id": path.stem,
        "audio_path": str(path),
        "text": text,
        "language": getattr(info, "language", config.transcription.language),
        "duration": _audio_duration(path),
        "avg_logprob": _mean_optional(segment_items, "avg_logprob"),
        "no_speech_prob": _mean_optional(segment_items, "no_speech_prob"),
    }


def transcribe_all_segments(
    manifest_path: str | Path,
    config: AppConfig,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Transcribe all exported segments listed in an export manifest."""

    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise TranscriptionError(f"Export manifest file does not exist: {manifest_file}")
    if not manifest_file.is_file():
        raise TranscriptionError(f"Export manifest path is not a file: {manifest_file}")

    manifest = load_json(manifest_file)
    if not isinstance(manifest, list):
        raise TranscriptionError(f"Export manifest JSON must contain a list: {manifest_file}")

    transcript_writer = (
        _JsonlTranscriptWriter(output_path, expected_count=len(manifest))
        if output_path is not None
        else None
    )
    model = load_whisper_model(config)
    results: list[dict[str, Any]] = []

    try:
        if transcript_writer is not None:
            transcript_writer.open()

        for index, item in enumerate(manifest, start=1):
            if not isinstance(item, dict):
                logger.warning("Skipping manifest item %d because it is not an object", index)
                result = _failed_record(
                    index=index,
                    error="Manifest item is not an object.",
                )
                results.append(result)
                if transcript_writer is not None:
                    transcript_writer.write_record(result)
                continue

            segment_id = str(item.get("id", f"{index:06d}"))
            audio_path = Path(str(item.get("audio_path", "")))
            logger.info("Transcribing segment %s (%d/%d)", segment_id, index, len(manifest))

            try:
                result = transcribe_segment(audio_path, model, config)
                result["id"] = segment_id
                result["audio_path"] = str(audio_path)
                result["duration"] = float(item.get("duration", result["duration"]))
                result["error"] = None
            except Exception as exc:
                logger.warning("Failed to transcribe segment %s: %s", segment_id, exc)
                result = {
                    "id": segment_id,
                    "audio_path": str(audio_path),
                    "text": "",
                    "language": config.transcription.language,
                    "duration": _safe_float(item.get("duration")),
                    "avg_logprob": None,
                    "no_speech_prob": None,
                    "error": str(exc),
                }

            results.append(result)
            if transcript_writer is not None:
                transcript_writer.write_record(result)
    except Exception:
        if transcript_writer is not None:
            transcript_writer.abort()
        raise
    else:
        if transcript_writer is not None:
            transcript_writer.close()

    success_count = sum(1 for item in results if item.get("error") is None)
    logger.info(
        "Transcription complete: %d succeeded, %d failed",
        success_count,
        len(results) - success_count,
    )
    return results


def save_transcripts(results: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Save raw transcript records as JSON Lines."""

    writer = _JsonlTranscriptWriter(output_path, expected_count=len(results))
    try:
        writer.open()
        for item in results:
            writer.write_record(item)
        return writer.close()
    except Exception as exc:
        writer.abort()
        logger.exception("Failed to save raw transcripts to %s: %s", writer.path, exc)
        raise TranscriptionError(
            f"Failed to save raw transcripts to {writer.path}: {exc}"
        ) from exc


class _JsonlTranscriptWriter:
    def __init__(self, output_path: str | Path, expected_count: int) -> None:
        self.path = Path(output_path)
        self.temp_path = self.path.with_name(f"{self.path.name}.tmp")
        self.expected_count = expected_count
        self.written_count = 0
        self._file: Any | None = None

    def open(self) -> None:
        logger.info(
            "Saving %d raw transcript records to %s via temp file %s",
            self.expected_count,
            self.path,
            self.temp_path,
        )
        ensure_dir(self.path.parent)
        self._file = self.temp_path.open("w", encoding="utf-8", newline="\n")

    def write_record(self, record: dict[str, Any]) -> None:
        if self._file is None:
            raise TranscriptionError("Transcript writer is not open.")

        try:
            safe_record = _json_safe(record)
            self._file.write(json.dumps(safe_record, ensure_ascii=False))
            self._file.write("\n")
        except Exception as exc:
            record_id = record.get("id") if isinstance(record, dict) else None
            raise TranscriptionError(
                f"Failed to serialize transcript record {record_id!r}: {exc}"
            ) from exc

        self.written_count += 1
        if self.written_count == 1 or self.written_count % 100 == 0:
            logger.info(
                "Transcript JSONL write progress: %d/%d records",
                self.written_count,
                self.expected_count,
            )

    def close(self) -> Path:
        if self._file is None:
            raise TranscriptionError("Transcript writer is not open.")

        try:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            self._file = None

            logger.info(
                "Finalizing transcript JSONL: replacing %s with %s",
                self.path,
                self.temp_path,
            )
            self.temp_path.replace(self.path)
            self._validate_output()
        except Exception as exc:
            raise TranscriptionError(
                f"Failed to finalize transcript JSONL {self.path}: {exc}"
            ) from exc

        logger.info(
            "Saved raw transcripts: %s (%d records, %d bytes)",
            self.path,
            self.written_count,
            self.path.stat().st_size,
        )
        return self.path

    def abort(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
        if self.temp_path.exists():
            logger.warning(
                "Leaving partial transcript temp file for diagnostics: %s (%d records written)",
                self.temp_path,
                self.written_count,
            )

    def _validate_output(self) -> None:
        if not self.path.exists():
            raise TranscriptionError(f"Transcript output file was not created: {self.path}")
        if self.expected_count > 0 and self.path.stat().st_size == 0:
            raise TranscriptionError(f"Transcript output file is empty after save: {self.path}")
        line_count = _count_lines(self.path)
        if line_count != self.written_count:
            raise TranscriptionError(
                f"Transcript output line count mismatch for {self.path}: "
                f"expected {self.written_count}, got {line_count}."
            )


def _resolve_device(requested_device: str) -> str:
    requested = requested_device.strip().lower()
    if requested == "auto":
        return "cuda" if _cuda_available() else "cpu"
    if requested == "cuda" and not _cuda_available():
        logger.warning("CUDA was requested but is unavailable. Falling back to CPU.")
        return "cpu"
    if requested not in {"cpu", "cuda"}:
        logger.warning("Unsupported transcription device '%s'. Falling back to CPU.", requested)
        return "cpu"
    return requested


def _resolve_compute_type(requested_compute_type: str, device: str) -> str:
    compute_type = requested_compute_type.strip().lower()
    if device == "cpu" and compute_type in {"float16", "bfloat16"}:
        logger.warning(
            "compute_type=%s is not suitable for CPU. Falling back to int8.",
            compute_type,
        )
        return "int8"
    return compute_type


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        pass

    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _audio_duration(path: Path) -> float | None:
    try:
        import soundfile as sf

        info = sf.info(path)
        return round(float(info.frames) / float(info.samplerate), 3)
    except Exception:
        return None


def _mean_optional(items: list[Any], attribute: str) -> float | None:
    values = [
        float(value)
        for value in (getattr(item, attribute, None) for item in items)
        if value is not None
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def _failed_record(index: int, error: str) -> dict[str, Any]:
    return {
        "id": f"{index:06d}",
        "audio_path": "",
        "text": "",
        "language": None,
        "duration": None,
        "avg_logprob": None,
        "no_speech_prob": None,
        "error": error,
    }
