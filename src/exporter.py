"""Export audio segment WAV files and final dataset metadata."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.utils import ensure_dir, load_json, save_json


logger = logging.getLogger("vn-audio2dataset.exporter")

DEFAULT_DURATION_TOLERANCE_SEC = 0.05


class ExporterError(RuntimeError):
    """Raised when audio segment export cannot complete."""


def load_final_segments(path: str | Path) -> list[dict[str, Any]]:
    """Load final segment metadata from JSON."""

    input_path = Path(path)
    if not input_path.exists():
        raise ExporterError(f"Final segments file does not exist: {input_path}")
    if not input_path.is_file():
        raise ExporterError(f"Final segments path is not a file: {input_path}")

    try:
        raw_data = load_json(input_path)
    except Exception as exc:
        raise ExporterError(
            f"Failed to read final segments from {input_path}: {exc}"
        ) from exc

    if not isinstance(raw_data, list):
        raise ExporterError(f"Final segments JSON must contain a list: {input_path}")

    return [_parse_segment(item, index) for index, item in enumerate(raw_data)]


def cut_audio_segments(
    audio_path: str | Path,
    segments: list[dict[str, Any]],
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    """Cut final segments from a master WAV file and return manifest items."""

    try:
        import soundfile as sf
    except ImportError as exc:
        raise ExporterError(
            "soundfile is not installed. Run 'pip install -r requirements.txt' "
            "and try again."
        ) from exc

    master_path = Path(audio_path)
    if not master_path.exists():
        raise ExporterError(f"Master audio file does not exist: {master_path}")
    if not master_path.is_file():
        raise ExporterError(f"Master audio path is not a file: {master_path}")

    wav_dir = ensure_dir(output_dir)
    manifest: list[dict[str, Any]] = []

    try:
        with sf.SoundFile(master_path, mode="r") as source:
            sample_rate = int(source.samplerate)
            total_frames = int(len(source))
            subtype = source.subtype

            for index, raw_segment in enumerate(segments, start=1):
                segment = _parse_segment(raw_segment, index - 1)
                start_frame, end_frame = _segment_to_frames(
                    segment,
                    sample_rate,
                    total_frames,
                )
                expected_duration = (end_frame - start_frame) / sample_rate
                output_path = wav_dir / f"{segment['id']}.wav"

                source.seek(start_frame)
                audio = source.read(end_frame - start_frame, dtype="float32")
                sf.write(output_path, audio, sample_rate, subtype=subtype)

                validate_exported_segment(
                    output_path,
                    expected_duration=expected_duration,
                    tolerance=DEFAULT_DURATION_TOLERANCE_SEC,
                )

                item = {
                    "id": segment["id"],
                    "audio_path": str(output_path),
                    "start": round(segment["start"], 3),
                    "end": round(segment["end"], 3),
                    "duration": round(expected_duration, 3),
                }
                manifest.append(item)

                if index <= 10 or index == len(segments) or index % 25 == 0:
                    logger.info("Exported segment %s -> %s", segment["id"], output_path)
                else:
                    logger.debug("Exported segment %s -> %s", segment["id"], output_path)
    except ExporterError:
        raise
    except Exception as exc:
        raise ExporterError(f"Failed to export audio segments from {master_path}: {exc}") from exc

    logger.info("Exported %d audio segments to %s", len(manifest), wav_dir)
    return manifest


def validate_exported_segment(
    path: str | Path,
    expected_duration: float,
    tolerance: float = DEFAULT_DURATION_TOLERANCE_SEC,
) -> None:
    """Validate an exported WAV duration against the expected duration."""

    try:
        import soundfile as sf
    except ImportError as exc:
        raise ExporterError(
            "soundfile is not installed. Run 'pip install -r requirements.txt' "
            "and try again."
        ) from exc

    output_path = Path(path)
    if not output_path.exists():
        raise ExporterError(f"Exported segment file does not exist: {output_path}")

    try:
        info = sf.info(output_path)
    except Exception as exc:
        raise ExporterError(f"Failed to inspect exported segment {output_path}: {exc}") from exc

    actual_duration = float(info.frames) / float(info.samplerate)
    delta = abs(actual_duration - expected_duration)
    if delta > tolerance:
        raise ExporterError(
            f"Exported segment duration mismatch for {output_path}: "
            f"expected {expected_duration:.3f}s, got {actual_duration:.3f}s "
            f"(tolerance {tolerance:.3f}s)."
        )


def export_metadata_csv(accepted_path: str | Path, output_path: str | Path) -> int:
    """Export metadata.csv in filename|text format from accepted records."""

    source_path = Path(accepted_path)
    target_path = Path(output_path)
    temp_path = target_path.with_name(f"{target_path.name}.tmp")
    count = 0

    logger.info("Exporting metadata.csv from %s to %s", source_path, target_path)
    try:
        ensure_dir(target_path.parent)
        with source_path.open("r", encoding="utf-8-sig") as input_file, temp_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as output_file:
            for record in _iter_accepted_records(input_file, source_path):
                audio_name = Path(str(record["audio_path"])).name
                text = _metadata_text(str(record["cleaned_text"]))
                output_file.write(f"{audio_name}|{text}\n")
                count += 1
                if count == 1 or count % 1000 == 0:
                    logger.info("metadata.csv export rows: %d", count)

            output_file.flush()
            os.fsync(output_file.fileno())

        temp_path.replace(target_path)
    except Exception as exc:
        raise ExporterError(f"Failed to export metadata.csv to {target_path}: {exc}") from exc

    logger.info("Saved metadata.csv: %s (%d rows)", target_path, count)
    return count


def export_manifest_jsonl(accepted_path: str | Path, output_path: str | Path) -> int:
    """Export final manifest.jsonl from accepted records."""

    source_path = Path(accepted_path)
    target_path = Path(output_path)
    temp_path = target_path.with_name(f"{target_path.name}.tmp")
    count = 0

    logger.info("Exporting manifest.jsonl from %s to %s", source_path, target_path)
    try:
        ensure_dir(target_path.parent)
        with source_path.open("r", encoding="utf-8-sig") as input_file, temp_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as output_file:
            for record in _iter_accepted_records(input_file, source_path):
                item = {
                    "id": str(record["id"]),
                    "audio_filepath": _manifest_audio_path(
                        Path(str(record["audio_path"])),
                        target_path.parent,
                    ),
                    "text": str(record["cleaned_text"]),
                    "duration": float(record["duration"]),
                }
                output_file.write(json.dumps(item, ensure_ascii=False))
                output_file.write("\n")
                count += 1
                if count == 1 or count % 1000 == 0:
                    logger.info("manifest.jsonl export rows: %d", count)

            output_file.flush()
            os.fsync(output_file.fileno())

        temp_path.replace(target_path)
    except Exception as exc:
        raise ExporterError(f"Failed to export manifest.jsonl to {target_path}: {exc}") from exc

    logger.info("Saved manifest.jsonl: %s (%d rows)", target_path, count)
    return count


def generate_dataset_summary(accepted_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Generate summary.json from accepted records."""

    source_path = Path(accepted_path)
    target_path = Path(output_path)
    durations: list[float] = []

    logger.info("Generating dataset summary from %s to %s", source_path, target_path)
    try:
        with source_path.open("r", encoding="utf-8-sig") as input_file:
            for record in _iter_accepted_records(input_file, source_path):
                durations.append(float(record["duration"]))
    except Exception as exc:
        raise ExporterError(f"Failed to generate summary from {source_path}: {exc}") from exc

    total_duration = sum(durations)
    total_accepted = len(durations)
    summary = {
        "total_accepted": total_accepted,
        "total_duration_seconds": round(total_duration, 3),
        "total_duration_hours": round(total_duration / 3600.0, 4),
        "average_duration_seconds": round(total_duration / total_accepted, 3)
        if total_accepted
        else 0.0,
        "min_duration_seconds": round(min(durations), 3) if durations else 0.0,
        "max_duration_seconds": round(max(durations), 3) if durations else 0.0,
    }
    save_json(summary, target_path)
    logger.info("Saved summary.json: %s", target_path)
    return summary


def export_dataset_from_accepted(
    accepted_path: str | Path,
    metadata_path: str | Path,
    manifest_path: str | Path,
    summary_path: str | Path,
    config: AppConfig,
) -> dict[str, Any]:
    """Export final dataset metadata files from accepted records."""

    del config
    source_path = Path(accepted_path)
    if not source_path.exists():
        raise ExporterError(f"Accepted file does not exist: {source_path}")
    if not source_path.is_file():
        raise ExporterError(f"Accepted path is not a file: {source_path}")

    metadata_count = export_metadata_csv(source_path, metadata_path)
    manifest_count = export_manifest_jsonl(source_path, manifest_path)
    summary = generate_dataset_summary(source_path, summary_path)

    if metadata_count != manifest_count or metadata_count != summary["total_accepted"]:
        raise ExporterError(
            "Final dataset export count mismatch: "
            f"metadata={metadata_count}, manifest={manifest_count}, "
            f"summary={summary['total_accepted']}"
        )

    logger.info(
        "Final dataset export complete: %d rows, %.3f seconds",
        metadata_count,
        summary["total_duration_seconds"],
    )
    return {
        "total_accepted": metadata_count,
        "metadata_path": str(metadata_path),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        **summary,
    }


def _parse_segment(item: Any, index: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ExporterError(f"Final segment at index {index} must be an object.")

    try:
        segment_id = str(item["id"])
        start = float(item["start"])
        end = float(item["end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExporterError(
            f"Final segment at index {index} must contain id, start, and end."
        ) from exc

    if not segment_id:
        raise ExporterError(f"Final segment at index {index} has an empty id.")
    if start < 0:
        raise ExporterError(f"Final segment {segment_id} has negative start.")
    if end <= start:
        raise ExporterError(f"Final segment {segment_id} must end after it starts.")

    return {
        "id": segment_id,
        "start": start,
        "end": end,
        "duration": end - start,
    }


def _segment_to_frames(
    segment: dict[str, Any],
    sample_rate: int,
    total_frames: int,
) -> tuple[int, int]:
    start_frame = round(segment["start"] * sample_rate)
    end_frame = round(segment["end"] * sample_rate)

    if start_frame < 0:
        raise ExporterError(f"Segment {segment['id']} starts before the audio begins.")
    if end_frame <= start_frame:
        raise ExporterError(f"Segment {segment['id']} has an invalid frame range.")
    if end_frame > total_frames:
        audio_duration = total_frames / sample_rate
        raise ExporterError(
            f"Segment {segment['id']} ends after the master audio duration: "
            f"segment end {segment['end']:.3f}s, audio duration {audio_duration:.3f}s."
        )

    return start_frame, end_frame


def _iter_accepted_records(input_file: Any, source_path: Path) -> Any:
    for line_number, line in enumerate(input_file, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExporterError(f"Invalid JSONL at {source_path}:{line_number}: {exc}") from exc
        yield _parse_accepted_record(record, line_number, source_path)


def _parse_accepted_record(
    record: Any,
    line_number: int,
    source_path: Path,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ExporterError(f"Accepted record at {source_path}:{line_number} must be an object.")
    if record.get("accepted") is False:
        raise ExporterError(
            f"Accepted record at {source_path}:{line_number} has accepted=false."
        )

    try:
        record_id = str(record["id"])
        audio_path = str(record["audio_path"])
        cleaned_text = str(record["cleaned_text"])
        duration = float(record["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExporterError(
            f"Accepted record at {source_path}:{line_number} must contain "
            "id, audio_path, cleaned_text, and duration."
        ) from exc

    if not record_id:
        raise ExporterError(f"Accepted record at {source_path}:{line_number} has empty id.")
    if not audio_path:
        raise ExporterError(f"Accepted record {record_id} has empty audio_path.")
    if cleaned_text == "":
        raise ExporterError(f"Accepted record {record_id} has empty cleaned_text.")
    if duration <= 0:
        raise ExporterError(f"Accepted record {record_id} has invalid duration: {duration}.")

    return {
        "id": record_id,
        "audio_path": audio_path,
        "cleaned_text": cleaned_text,
        "duration": duration,
    }


def _metadata_text(text: str) -> str:
    return text.replace("\r", " ").replace("\n", " ").replace("|", " ").strip()


def _manifest_audio_path(audio_path: Path, base_dir: Path) -> str:
    try:
        return str(audio_path.relative_to(base_dir))
    except ValueError:
        try:
            return str(audio_path.resolve().relative_to(base_dir.resolve()))
        except ValueError:
            return str(audio_path)
