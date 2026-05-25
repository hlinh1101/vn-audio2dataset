"""Consolidate selected processed output folders into one fine-tuning dataset."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.exporter import ExporterError, export_dataset_from_accepted
from src.utils import ensure_dir, save_json


logger = logging.getLogger("vn-audio2dataset.consolidator")

_TOKEN_RE = re.compile(r"[^A-Za-z0-9._-]+")


class ConsolidationError(RuntimeError):
    """Raised when dataset consolidation cannot complete."""


def consolidate_output_folders(
    source_dirs: list[str | Path],
    output_dir: str | Path,
    config: AppConfig,
) -> dict[str, Any]:
    """Merge explicitly selected processed output folders into one dataset.

    Each source directory must contain an ``accepted.jsonl`` produced by the
    normal pipeline or export stage. Audio files referenced by accepted records
    are copied into ``output_dir/wavs`` with source-prefixed names to avoid
    collisions, then final metadata files are regenerated for the combined set.
    """

    sources = _validate_source_dirs(source_dirs)
    if not sources:
        raise ConsolidationError("At least one source output folder is required.")

    target_dir = Path(output_dir)
    _validate_target_dir(target_dir, sources)

    wav_dir = ensure_dir(target_dir / "wavs")
    accepted_path = target_dir / "accepted.jsonl"
    metadata_path = target_dir / "metadata.csv"
    manifest_path = target_dir / "manifest.jsonl"
    summary_path = target_dir / "summary.json"
    report_path = target_dir / "consolidation_report.json"

    logger.info("Consolidating %d source folders into %s", len(sources), target_dir)

    used_names: set[str] = set()
    used_ids: set[str] = set()
    used_source_labels: set[str] = set()
    source_reports: list[dict[str, Any]] = []
    total_records = 0
    accepted_tmp = accepted_path.with_name(f"{accepted_path.name}.tmp")

    try:
        ensure_dir(target_dir)
        with accepted_tmp.open("w", encoding="utf-8", newline="\n") as output_file:
            for source_index, source_dir in enumerate(sources, start=1):
                source_label = _unique_token(
                    _source_label(source_dir, source_index),
                    used_source_labels,
                )
                source_accepted_path = source_dir / "accepted.jsonl"
                source_count = 0
                source_duration = 0.0

                logger.info("Reading accepted records from %s", source_accepted_path)
                with source_accepted_path.open("r", encoding="utf-8-sig") as input_file:
                    for line_number, line in enumerate(input_file, start=1):
                        if not line.strip():
                            continue
                        record = _parse_accepted_record(
                            line,
                            source_accepted_path,
                            line_number,
                        )
                        audio_path = _resolve_audio_path(
                            str(record["audio_path"]),
                            source_dir,
                        )
                        copied_audio_path = _copy_audio_file(
                            audio_path=audio_path,
                            wav_dir=wav_dir,
                            source_label=source_label,
                            used_names=used_names,
                        )

                        merged_id = _unique_token(
                            f"{source_label}_{_safe_token(str(record['id']))}",
                            used_ids,
                        )
                        merged_record = {
                            **record,
                            "id": merged_id,
                            "audio_path": str(copied_audio_path),
                            "source_folder": str(source_dir),
                            "source_id": str(record["id"]),
                            "source_audio_path": str(audio_path),
                        }
                        output_file.write(json.dumps(merged_record, ensure_ascii=False))
                        output_file.write("\n")

                        source_count += 1
                        source_duration += float(record["duration"])
                        total_records += 1
                        if total_records == 1 or total_records % 1000 == 0:
                            logger.info(
                                "Consolidated accepted rows: %d",
                                total_records,
                            )

                source_reports.append(
                    {
                        "source_folder": str(source_dir),
                        "accepted_path": str(source_accepted_path),
                        "row_count": source_count,
                        "duration_seconds": round(source_duration, 3),
                    }
                )

            output_file.flush()
            os.fsync(output_file.fileno())
        accepted_tmp.replace(accepted_path)
    except ConsolidationError:
        raise
    except Exception as exc:
        raise ConsolidationError(f"Failed to consolidate into {target_dir}: {exc}") from exc

    if total_records == 0:
        raise ConsolidationError("Selected source folders contain 0 accepted records.")

    try:
        export_stats = export_dataset_from_accepted(
            accepted_path=accepted_path,
            metadata_path=metadata_path,
            manifest_path=manifest_path,
            summary_path=summary_path,
            config=config,
        )
    except ExporterError as exc:
        raise ConsolidationError(f"Failed to export consolidated dataset: {exc}") from exc

    report = {
        "source_count": len(sources),
        "sources": source_reports,
        "total_accepted": export_stats["total_accepted"],
        "total_duration_seconds": export_stats["total_duration_seconds"],
        "total_duration_hours": export_stats["total_duration_hours"],
        "output_dir": str(target_dir),
        "wavs_dir": str(wav_dir),
        "accepted_path": str(accepted_path),
        "metadata_path": str(metadata_path),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
    }
    save_json(report, report_path)

    logger.info(
        "Consolidated dataset complete: %d rows, %.3f seconds",
        export_stats["total_accepted"],
        export_stats["total_duration_seconds"],
    )
    return {
        **report,
        "report_path": str(report_path),
    }


def _validate_source_dir(path: str | Path) -> Path:
    source_dir = Path(path)
    if not source_dir.exists():
        raise ConsolidationError(f"Source output folder does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise ConsolidationError(f"Source output path is not a folder: {source_dir}")

    accepted_path = source_dir / "accepted.jsonl"
    if not accepted_path.exists():
        raise ConsolidationError(f"Source folder is missing accepted.jsonl: {source_dir}")
    if not accepted_path.is_file():
        raise ConsolidationError(f"accepted.jsonl is not a file: {accepted_path}")

    return source_dir


def _validate_source_dirs(paths: list[str | Path]) -> list[Path]:
    source_dirs: list[Path] = []
    seen_resolved: set[Path] = set()
    for path in paths:
        source_dir = _validate_source_dir(path)
        try:
            resolved = source_dir.resolve()
        except OSError as exc:
            raise ConsolidationError(f"Invalid source folder {source_dir}: {exc}") from exc
        if resolved in seen_resolved:
            raise ConsolidationError(f"Duplicate source output folder: {source_dir}")
        seen_resolved.add(resolved)
        source_dirs.append(source_dir)
    return source_dirs


def _validate_target_dir(target_dir: Path, source_dirs: list[Path]) -> None:
    try:
        target_resolved = target_dir.resolve()
    except OSError as exc:
        raise ConsolidationError(f"Invalid output folder {target_dir}: {exc}") from exc

    for source_dir in source_dirs:
        try:
            source_resolved = source_dir.resolve()
        except OSError as exc:
            raise ConsolidationError(f"Invalid source folder {source_dir}: {exc}") from exc
        if target_resolved == source_resolved:
            raise ConsolidationError(
                "Consolidation output folder must be different from every source folder: "
                f"{target_dir}"
            )
        if _is_relative_to(target_resolved, source_resolved):
            raise ConsolidationError(
                "Consolidation output folder must not be inside a source folder: "
                f"{target_dir}"
            )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _parse_accepted_record(line: str, source_path: Path, line_number: int) -> dict[str, Any]:
    try:
        raw_record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ConsolidationError(f"Invalid JSONL at {source_path}:{line_number}: {exc}") from exc

    if not isinstance(raw_record, dict):
        raise ConsolidationError(
            f"Accepted record at {source_path}:{line_number} must be an object."
        )
    if raw_record.get("accepted") is False:
        raise ConsolidationError(
            f"Accepted record at {source_path}:{line_number} has accepted=false."
        )

    try:
        record_id = str(raw_record["id"])
        audio_path = str(raw_record["audio_path"])
        cleaned_text = str(raw_record["cleaned_text"])
        duration = float(raw_record["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConsolidationError(
            f"Accepted record at {source_path}:{line_number} must contain "
            "id, audio_path, cleaned_text, and duration."
        ) from exc

    if not record_id:
        raise ConsolidationError(f"Accepted record at {source_path}:{line_number} has empty id.")
    if not audio_path:
        raise ConsolidationError(f"Accepted record {record_id} has empty audio_path.")
    if cleaned_text == "":
        raise ConsolidationError(f"Accepted record {record_id} has empty cleaned_text.")
    if duration <= 0:
        raise ConsolidationError(f"Accepted record {record_id} has invalid duration: {duration}.")

    return {
        **raw_record,
        "id": record_id,
        "audio_path": audio_path,
        "cleaned_text": cleaned_text,
        "duration": duration,
    }


def _resolve_audio_path(raw_audio_path: str, source_dir: Path) -> Path:
    audio_path = Path(raw_audio_path)
    candidates = [audio_path]
    if not audio_path.is_absolute():
        candidates.extend(
            [
                source_dir / audio_path,
                source_dir / "wavs" / audio_path.name,
            ]
        )

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    raise ConsolidationError(
        f"Audio file does not exist for source {source_dir}: {raw_audio_path}"
    )


def _copy_audio_file(
    audio_path: Path,
    wav_dir: Path,
    source_label: str,
    used_names: set[str],
) -> Path:
    suffix = audio_path.suffix or ".wav"
    base_name = _safe_token(audio_path.stem)
    candidate_name = f"{source_label}_{base_name}{suffix}"
    index = 2
    while candidate_name in used_names:
        candidate_name = f"{source_label}_{base_name}_{index}{suffix}"
        index += 1

    used_names.add(candidate_name)
    target_path = wav_dir / candidate_name
    shutil.copy2(audio_path, target_path)
    return target_path


def _source_label(source_dir: Path, source_index: int) -> str:
    label = _safe_token(source_dir.name)
    return label or f"source_{source_index:03d}"


def _safe_token(value: str) -> str:
    safe = _TOKEN_RE.sub("_", value.strip())
    safe = safe.strip("._-")
    return safe or "item"


def _unique_token(value: str, used_tokens: set[str]) -> str:
    token = value
    index = 2
    while token in used_tokens:
        token = f"{value}_{index}"
        index += 1
    used_tokens.add(token)
    return token
