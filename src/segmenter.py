"""Build final candidate segment metadata from raw VAD spans."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.config import AppConfig, SegmentConfig
from src.utils import load_json, save_json


logger = logging.getLogger("vn-audio2dataset.segmenter")


class SegmenterError(RuntimeError):
    """Raised when final segment metadata cannot be built."""


def load_vad_segments(path: str | Path) -> list[dict[str, float]]:
    """Load raw VAD segments from JSON."""

    input_path = Path(path)
    if not input_path.exists():
        raise SegmenterError(f"VAD segments file does not exist: {input_path}")
    if not input_path.is_file():
        raise SegmenterError(f"VAD segments path is not a file: {input_path}")

    try:
        raw_data = load_json(input_path)
    except Exception as exc:
        raise SegmenterError(f"Failed to read VAD segments from {input_path}: {exc}") from exc

    if not isinstance(raw_data, list):
        raise SegmenterError(f"VAD segments JSON must contain a list: {input_path}")

    segments = [_parse_vad_segment(item, index) for index, item in enumerate(raw_data)]
    segments.sort(key=lambda item: item["start"])
    return segments


def build_segments(
    vad_segments: list[dict[str, float]],
    config: AppConfig,
) -> list[dict[str, Any]]:
    """Build final candidate TTS segment metadata from raw VAD spans."""

    _validate_segment_config(config.segments)
    normalized = [_parse_vad_segment(item, index) for index, item in enumerate(vad_segments)]
    normalized.sort(key=lambda item: item["start"])

    merged = merge_short_segments(normalized, config)
    split = split_long_segments(merged, config)
    accepted = [
        item
        for item in split
        if config.segments.min_sec <= item["duration"] <= config.segments.max_sec
    ]
    final_segments = [_with_id(item, index) for index, item in enumerate(accepted, start=1)]

    logger.info(
        "Built %d final segments from %d VAD spans",
        len(final_segments),
        len(vad_segments),
    )
    return final_segments


def merge_short_segments(
    vad_segments: list[dict[str, float]],
    config: AppConfig,
) -> list[dict[str, Any]]:
    """Merge short neighboring VAD spans when their silence gap is small enough."""

    params = config.segments
    merged: list[dict[str, Any]] = []
    index = 0

    while index < len(vad_segments):
        current = _candidate_from_span(vad_segments[index], source="vad_raw")
        used_merge = False

        while (
            current["duration"] < params.min_sec
            and index + 1 < len(vad_segments)
            and _gap(current, vad_segments[index + 1]) <= params.merge_gap_sec
            and _combined_duration(current, vad_segments[index + 1]) <= params.max_sec
        ):
            index += 1
            current = _merge_pair(current, vad_segments[index])
            used_merge = True

            if params.ideal_min <= current["duration"] <= params.ideal_max:
                break

        if used_merge:
            current["source"] = "vad_merged"

        merged.append(current)
        index += 1

    return merged


def split_long_segments(
    segments: list[dict[str, Any]],
    config: AppConfig,
) -> list[dict[str, Any]]:
    """Split spans longer than max_sec using deterministic duration-based chunks."""

    params = config.segments
    output: list[dict[str, Any]] = []

    for segment in segments:
        if segment["duration"] <= params.max_sec:
            output.append(segment)
            continue

        output.extend(_fallback_split(segment, params))

    return output


def save_final_segments(
    segments: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Save final segment metadata as JSON."""

    saved_path = save_json(segments, output_path)
    logger.info("Saved final segments: %s", saved_path)
    return saved_path


def _parse_vad_segment(item: Any, index: int) -> dict[str, float]:
    if not isinstance(item, dict):
        raise SegmenterError(f"VAD segment at index {index} must be an object.")

    try:
        start = float(item["start"])
        end = float(item["end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SegmenterError(
            f"VAD segment at index {index} must contain numeric start and end."
        ) from exc

    if start < 0:
        raise SegmenterError(f"VAD segment at index {index} has negative start.")
    if end <= start:
        raise SegmenterError(f"VAD segment at index {index} must end after it starts.")

    return {
        "start": start,
        "end": end,
        "duration": end - start,
    }


def _validate_segment_config(config: SegmentConfig) -> None:
    if config.min_sec <= 0:
        raise SegmenterError("segments.min_sec must be greater than zero.")
    if config.max_sec < config.min_sec:
        raise SegmenterError("segments.max_sec must be greater than or equal to min_sec.")
    if config.ideal_min < config.min_sec:
        raise SegmenterError("segments.ideal_min must be greater than or equal to min_sec.")
    if config.ideal_max > config.max_sec:
        raise SegmenterError("segments.ideal_max must be less than or equal to max_sec.")
    if config.ideal_max < config.ideal_min:
        raise SegmenterError("segments.ideal_max must be greater than or equal to ideal_min.")
    if config.merge_gap_sec < 0:
        raise SegmenterError("segments.merge_gap_sec cannot be negative.")
    if config.force_split_target_sec <= 0:
        raise SegmenterError("segments.force_split_target_sec must be greater than zero.")
    if config.force_split_target_sec > config.max_sec:
        raise SegmenterError(
            "segments.force_split_target_sec must be less than or equal to max_sec."
        )


def _candidate_from_span(span: dict[str, float], source: str) -> dict[str, Any]:
    return {
        "start": span["start"],
        "end": span["end"],
        "duration": span["end"] - span["start"],
        "source": source,
    }


def _gap(left: dict[str, Any], right: dict[str, float]) -> float:
    return max(0.0, right["start"] - left["end"])


def _combined_duration(left: dict[str, Any], right: dict[str, float]) -> float:
    return right["end"] - left["start"]


def _merge_pair(left: dict[str, Any], right: dict[str, float]) -> dict[str, Any]:
    start = left["start"]
    end = right["end"]
    return {
        "start": start,
        "end": end,
        "duration": end - start,
        "source": "vad_merged",
    }


def _fallback_split(
    segment: dict[str, Any],
    config: SegmentConfig,
) -> list[dict[str, Any]]:
    start = float(segment["start"])
    end = float(segment["end"])
    duration = end - start
    chunk_count = max(2, int(duration / config.force_split_target_sec + 0.999999))

    while duration / chunk_count > config.max_sec:
        chunk_count += 1

    chunk_duration = duration / chunk_count
    chunks: list[dict[str, Any]] = []

    for index in range(chunk_count):
        chunk_start = start + (chunk_duration * index)
        chunk_end = end if index == chunk_count - 1 else start + (chunk_duration * (index + 1))
        chunks.append(
            {
                "start": chunk_start,
                "end": chunk_end,
                "duration": chunk_end - chunk_start,
                "source": "vad_split",
            }
        )

    return chunks


def _with_id(segment: dict[str, Any], index: int) -> dict[str, Any]:
    start = round(float(segment["start"]), 3)
    end = round(float(segment["end"]), 3)
    duration = round(max(0.0, end - start), 3)
    return {
        "id": f"{index:06d}",
        "start": start,
        "end": end,
        "duration": duration,
        "source": str(segment["source"]),
    }
