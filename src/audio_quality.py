"""Rule-based audio quality gating for exported segment WAV files."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from src.config import AppConfig
from src.utils import load_json, save_json

import logging


logger = logging.getLogger("vn-audio2dataset.audio_quality")


class AudioQualityError(RuntimeError):
    """Raised when audio quality classification cannot complete."""


def load_audio_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load an exported audio manifest."""

    manifest_path = Path(path)
    if not manifest_path.exists():
        raise AudioQualityError(f"Audio manifest does not exist: {manifest_path}")
    if not manifest_path.is_file():
        raise AudioQualityError(f"Audio manifest path is not a file: {manifest_path}")

    try:
        raw = load_json(manifest_path)
    except Exception as exc:
        raise AudioQualityError(
            f"Failed to read audio manifest {manifest_path}: {exc}"
        ) from exc

    if not isinstance(raw, list):
        raise AudioQualityError(f"Audio manifest must contain a list: {manifest_path}")

    items: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        items.append(_parse_manifest_record(item, manifest_path, index))
    return items


def filter_audio_manifest(
    manifest_path: str | Path,
    good_output_path: str | Path,
    bad_output_path: str | Path,
    review_output_path: str | Path,
    report_output_path: str | Path,
    config: AppConfig,
) -> dict[str, Any]:
    """Classify exported audio segments into good, bad, and review groups."""

    records = load_audio_manifest(manifest_path)
    logger.info("Running audio quality analysis on %d segments", len(records))

    good_items: list[dict[str, Any]] = []
    bad_items: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, float]] = []
    bad_reason_counts: dict[str, int] = {}
    review_reason_counts: dict[str, int] = {}

    for index, record in enumerate(records, start=1):
        item = evaluate_audio_segment(record, config)
        label = str(item["quality_label"])

        if label == "good":
            good_items.append(item)
        elif label == "bad":
            bad_items.append(item)
            for reason in item["quality_reasons"]:
                bad_reason_counts[reason] = bad_reason_counts.get(reason, 0) + 1
        else:
            review_items.append(item)
            for reason in item["quality_reasons"]:
                review_reason_counts[reason] = review_reason_counts.get(reason, 0) + 1

        metrics = item.get("quality_metrics", {})
        if isinstance(metrics, dict):
            numeric_metrics = {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            }
            if numeric_metrics:
                metrics_rows.append(numeric_metrics)

        if index == 1 or index % 250 == 0:
            logger.info(
                "Audio quality progress: %d/%d (good=%d, review=%d, bad=%d)",
                index,
                len(records),
                len(good_items),
                len(review_items),
                len(bad_items),
            )

    save_json(good_items, good_output_path)
    save_json(bad_items, bad_output_path)
    save_json(review_items, review_output_path)

    report = {
        "total_segments": len(records),
        "good_count": len(good_items),
        "review_count": len(review_items),
        "bad_count": len(bad_items),
        "good_duration_seconds": round(_total_duration(good_items), 3),
        "review_duration_seconds": round(_total_duration(review_items), 3),
        "bad_duration_seconds": round(_total_duration(bad_items), 3),
        "bad_reason_counts": dict(sorted(bad_reason_counts.items())),
        "review_reason_counts": dict(sorted(review_reason_counts.items())),
        "thresholds": {
            "min_sec": config.audio_quality.min_sec,
            "max_sec": config.audio_quality.max_sec,
            "silence_threshold_dbfs": config.audio_quality.silence_threshold_dbfs,
            "min_rms_dbfs": config.audio_quality.min_rms_dbfs,
            "review_min_rms_dbfs": config.audio_quality.review_min_rms_dbfs,
            "max_silence_ratio": config.audio_quality.max_silence_ratio,
            "review_silence_ratio": config.audio_quality.review_silence_ratio,
            "max_leading_silence_sec": config.audio_quality.max_leading_silence_sec,
            "review_leading_silence_sec": config.audio_quality.review_leading_silence_sec,
            "max_trailing_silence_sec": config.audio_quality.max_trailing_silence_sec,
            "review_trailing_silence_sec": config.audio_quality.review_trailing_silence_sec,
            "max_clipping_ratio": config.audio_quality.max_clipping_ratio,
            "review_clipping_ratio": config.audio_quality.review_clipping_ratio,
            "max_spectral_flatness": config.audio_quality.max_spectral_flatness,
            "review_spectral_flatness": config.audio_quality.review_spectral_flatness,
            "max_high_freq_energy_ratio": config.audio_quality.max_high_freq_energy_ratio,
            "review_high_freq_energy_ratio": config.audio_quality.review_high_freq_energy_ratio,
        },
        "metric_summary": _metric_summary(metrics_rows),
    }
    save_json(report, report_output_path)

    logger.info("Saved audio quality good manifest: %s", good_output_path)
    logger.info("Saved audio quality review manifest: %s", review_output_path)
    logger.info("Saved audio quality bad manifest: %s", bad_output_path)
    logger.info("Saved audio quality report: %s", report_output_path)
    logger.info(
        "Audio quality summary: good=%d, review=%d, bad=%d",
        len(good_items),
        len(review_items),
        len(bad_items),
    )

    return report


def evaluate_audio_segment(record: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    """Analyze and classify one exported audio segment."""

    parsed = _parse_manifest_record(record, Path("<memory>"), 0)
    reasons_bad: list[str] = []
    reasons_review: list[str] = []

    duration = float(parsed["duration"])
    if duration < config.audio_quality.min_sec:
        reasons_bad.append("duration_too_short")
    if duration > config.audio_quality.max_sec:
        reasons_bad.append("duration_too_long")

    metrics = analyze_audio_file(parsed["audio_path"], config)
    if metrics.get("error") is not None:
        reasons_bad.append(str(metrics["error"]))
        label = "bad"
    else:
        _apply_quality_rules(metrics, reasons_bad, reasons_review, config)
        if reasons_bad:
            label = "bad"
        elif reasons_review:
            label = "review"
        else:
            label = "good"

    item = dict(parsed)
    item["quality_label"] = label
    item["quality_reasons"] = reasons_bad or reasons_review
    item["quality_metrics"] = metrics
    return item


def analyze_audio_file(audio_path: str | Path, config: AppConfig) -> dict[str, Any]:
    """Extract basic quality metrics from an audio segment."""
    path = Path(audio_path)
    if not path.exists():
        return {"error": "audio_missing"}
    if not path.is_file():
        return {"error": "audio_path_not_file"}

    try:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        logger.exception("Failed to read audio segment %s", path)
        return {"error": "audio_read_failed"}

    if audio.size == 0:
        return {"error": "audio_empty"}

    channels = int(audio.shape[1])
    mono = audio.mean(axis=1)
    abs_mono = np.abs(mono)
    silence_threshold = 10 ** (config.audio_quality.silence_threshold_dbfs / 20.0)
    voiced_mask = abs_mono >= silence_threshold

    lead_samples = 0
    while lead_samples < len(voiced_mask) and not voiced_mask[lead_samples]:
        lead_samples += 1

    trail_samples = 0
    while trail_samples < len(voiced_mask) and not voiced_mask[len(voiced_mask) - 1 - trail_samples]:
        trail_samples += 1

    rms = float(np.sqrt(np.mean(np.square(mono))))
    peak = float(abs_mono.max())
    clipping_ratio = float((abs_mono >= 0.999).mean())
    silence_ratio = float(1.0 - voiced_mask.mean())

    analysis_window = mono[: min(len(mono), sample_rate * 2)]
    if len(analysis_window) >= 16:
        spectrum = np.abs(
            np.fft.rfft(analysis_window * np.hanning(len(analysis_window)))
        ) + 1e-12
        freqs = np.fft.rfftfreq(len(analysis_window), 1.0 / sample_rate)
        spectral_flatness = float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))
        high_freq_energy_ratio = float(spectrum[freqs >= 5000].sum() / spectrum.sum())
    else:
        spectral_flatness = 0.0
        high_freq_energy_ratio = 0.0

    return {
        "sample_rate": float(sample_rate),
        "channels": float(channels),
        "rms_dbfs": round(_dbfs(rms), 3),
        "peak_dbfs": round(_dbfs(peak), 3),
        "clipping_ratio": round(clipping_ratio, 6),
        "silence_ratio": round(silence_ratio, 6),
        "leading_silence_sec": round(lead_samples / sample_rate, 4),
        "trailing_silence_sec": round(trail_samples / sample_rate, 4),
        "spectral_flatness": round(spectral_flatness, 6),
        "high_freq_energy_ratio": round(high_freq_energy_ratio, 6),
        "error": None,
    }


def _apply_quality_rules(
    metrics: dict[str, Any],
    reasons_bad: list[str],
    reasons_review: list[str],
    config: AppConfig,
) -> None:
    rms_dbfs = float(metrics["rms_dbfs"])
    silence_ratio = float(metrics["silence_ratio"])
    leading_silence_sec = float(metrics["leading_silence_sec"])
    trailing_silence_sec = float(metrics["trailing_silence_sec"])
    clipping_ratio = float(metrics["clipping_ratio"])
    spectral_flatness = float(metrics["spectral_flatness"])
    high_freq_energy_ratio = float(metrics["high_freq_energy_ratio"])

    if rms_dbfs < config.audio_quality.min_rms_dbfs:
        reasons_review.append("audio_too_quiet")
    elif rms_dbfs < config.audio_quality.review_min_rms_dbfs:
        reasons_review.append("audio_quiet_review")

    if clipping_ratio > config.audio_quality.max_clipping_ratio:
        reasons_bad.append("possible_clipping")
    elif clipping_ratio > config.audio_quality.review_clipping_ratio:
        reasons_review.append("possible_clipping_review")

    if silence_ratio > config.audio_quality.max_silence_ratio:
        reasons_bad.append("excessive_internal_silence")
    elif silence_ratio > config.audio_quality.review_silence_ratio:
        reasons_review.append("internal_silence_review")

    if leading_silence_sec > config.audio_quality.max_leading_silence_sec:
        reasons_bad.append("leading_silence_too_long")
    elif leading_silence_sec > config.audio_quality.review_leading_silence_sec:
        reasons_review.append("leading_silence_review")

    if trailing_silence_sec > config.audio_quality.max_trailing_silence_sec:
        reasons_bad.append("trailing_silence_too_long")
    elif trailing_silence_sec > config.audio_quality.review_trailing_silence_sec:
        reasons_review.append("trailing_silence_review")

    if spectral_flatness > config.audio_quality.max_spectral_flatness:
        reasons_bad.append("noise_or_reverb_proxy")
    elif spectral_flatness > config.audio_quality.review_spectral_flatness:
        reasons_review.append("noise_or_reverb_review")

    if high_freq_energy_ratio > config.audio_quality.max_high_freq_energy_ratio:
        reasons_bad.append("high_frequency_noise_proxy")
    elif high_freq_energy_ratio > config.audio_quality.review_high_freq_energy_ratio:
        reasons_review.append("high_frequency_noise_review")


def _parse_manifest_record(
    item: Any,
    source_path: Path,
    index: int,
) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise AudioQualityError(
            f"Manifest record at {source_path}:{index} must be an object."
        )

    try:
        record_id = str(item["id"])
        audio_path = str(item["audio_path"])
        duration = float(item["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AudioQualityError(
            f"Manifest record at {source_path}:{index} must contain id, audio_path, "
            "and duration."
        ) from exc

    record = {
        "id": record_id,
        "audio_path": audio_path,
        "duration": duration,
    }
    for key in ("start", "end"):
        if key in item:
            record[key] = item[key]
    return record


def _dbfs(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-12))


def _total_duration(items: list[dict[str, Any]]) -> float:
    return sum(float(item.get("duration", 0.0) or 0.0) for item in items)


def _metric_summary(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not rows:
        return {}

    keys = sorted({key for row in rows for key in row})
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = sorted(row[key] for row in rows if key in row)
        if not values:
            continue
        summary[key] = {
            "min": round(values[0], 6),
            "avg": round(sum(values) / len(values), 6),
            "p50": round(_percentile(values, 0.50), 6),
            "p90": round(_percentile(values, 0.90), 6),
            "p95": round(_percentile(values, 0.95), 6),
            "max": round(values[-1], 6),
        }
    return summary


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, int(round((len(values) - 1) * q)))
    return values[index]
