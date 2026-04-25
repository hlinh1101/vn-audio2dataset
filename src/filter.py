"""Rule-based quality filtering for cleaned transcript records."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.utils import ensure_dir, save_json


logger = logging.getLogger("vn-audio2dataset.filter")

_WORD_RE = re.compile(r"\S+")
_COMMON_TEXT_RE = re.compile(
    r"^[\w\s\u00c0-\u1ef9.,?!:;\-'\u201c\u201d\u2018\u2019`\u00b4]+$",
    re.UNICODE,
)


class FilterError(RuntimeError):
    """Raised when filtering cannot complete."""


def evaluate_audio_quality(
    audio_path: str | Path,
    record: dict[str, Any],
    config: AppConfig,
) -> dict[str, Any]:
    """Evaluate audio-related quality checks for one segment."""

    reasons: list[str] = []
    path = Path(audio_path)
    duration = _safe_float(record.get("duration"))

    if duration is None:
        reasons.append("missing_duration")
    else:
        if duration < config.filter.min_sec:
            reasons.append("duration_too_short")
        if duration > config.filter.max_sec:
            reasons.append("duration_too_long")

    if not path.exists():
        reasons.append("audio_missing")
        return {"reject_reasons": reasons, "rms": None}
    if not path.is_file():
        reasons.append("audio_path_not_file")
        return {"reject_reasons": reasons, "rms": None}

    rms = None
    if config.filter.min_rms is not None:
        try:
            rms = _compute_rms(path)
            if rms < config.filter.min_rms:
                reasons.append("audio_rms_too_low")
        except Exception as exc:
            reasons.append("audio_analysis_failed")
            logger.warning("Audio analysis failed for %s: %s", path, exc)

    return {"reject_reasons": reasons, "rms": rms}


def evaluate_text_quality(record: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    """Evaluate text and ASR-confidence checks for one cleaned transcript."""

    reasons: list[str] = []
    cleaned_text = str(record.get("cleaned_text") or "")
    word_count = len(_WORD_RE.findall(cleaned_text))
    char_count = len(cleaned_text)

    if cleaned_text == "":
        reasons.append("empty_cleaned_text")
    if bool(record.get("is_empty_after_cleaning")):
        reasons.append("marked_empty_after_cleaning")
    if word_count < config.filter.min_words:
        reasons.append("word_count_too_low")
    if char_count > config.filter.max_chars:
        reasons.append("char_count_too_high")
    if (
        config.filter.max_unusual_symbol_ratio is not None
        and cleaned_text
        and _unusual_symbol_ratio(cleaned_text) > config.filter.max_unusual_symbol_ratio
    ):
        reasons.append("too_many_unusual_symbols")

    avg_logprob = _safe_float(record.get("avg_logprob"))
    if (
        config.filter.min_avg_logprob is not None
        and avg_logprob is not None
        and avg_logprob < config.filter.min_avg_logprob
    ):
        reasons.append("avg_logprob_too_low")

    no_speech_prob = _safe_float(record.get("no_speech_prob"))
    if (
        config.filter.max_no_speech_prob is not None
        and no_speech_prob is not None
        and no_speech_prob > config.filter.max_no_speech_prob
    ):
        reasons.append("no_speech_prob_too_high")

    return {
        "reject_reasons": reasons,
        "word_count": word_count,
        "char_count": char_count,
    }


def filter_record(record: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    """Filter one cleaned transcript record and return a normalized decision item."""

    record_id = str(record.get("id", ""))
    audio_path = str(record.get("audio_path", ""))
    cleaned_text = str(record.get("cleaned_text") or "")
    duration = _safe_float(record.get("duration"))

    text_result = evaluate_text_quality(record, config)
    audio_result = evaluate_audio_quality(audio_path, record, config)
    reject_reasons = [
        *text_result["reject_reasons"],
        *audio_result["reject_reasons"],
    ]

    return {
        "id": record_id,
        "accepted": len(reject_reasons) == 0,
        "cleaned_text": cleaned_text,
        "audio_path": audio_path,
        "duration": duration,
        "reject_reasons": reject_reasons,
        "word_count": text_result["word_count"],
        "char_count": text_result["char_count"],
    }


def filter_all(
    cleaned_input_path: str | Path,
    accepted_output_path: str | Path,
    rejected_output_path: str | Path,
    report_output_path: str | Path,
    config: AppConfig,
) -> dict[str, int]:
    """Filter cleaned transcript records and save accepted, rejected, and report files."""

    input_path = Path(cleaned_input_path)
    accepted_path = Path(accepted_output_path)
    rejected_path = Path(rejected_output_path)
    report_path = Path(report_output_path)

    if not input_path.exists():
        raise FilterError(f"Cleaned transcripts file does not exist: {input_path}")
    if not input_path.is_file():
        raise FilterError(f"Cleaned transcripts path is not a file: {input_path}")

    ensure_dir(accepted_path.parent)
    ensure_dir(rejected_path.parent)
    ensure_dir(report_path.parent)

    accepted_tmp = accepted_path.with_name(f"{accepted_path.name}.tmp")
    rejected_tmp = rejected_path.with_name(f"{rejected_path.name}.tmp")
    total_count = 0
    accepted_count = 0
    rejected_count = 0
    reason_counts: dict[str, int] = {}

    logger.info("Filtering cleaned transcripts from %s", input_path)
    logger.info("Accepted output: %s", accepted_path)
    logger.info("Rejected output: %s", rejected_path)
    logger.info("Report output: %s", report_path)

    try:
        with input_path.open("r", encoding="utf-8-sig") as input_file, accepted_tmp.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as accepted_file, rejected_tmp.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as rejected_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise FilterError(
                        f"Invalid JSONL at {input_path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise FilterError(
                        f"Cleaned transcript record at {input_path}:{line_number} "
                        "must be an object."
                    )

                item = filter_record(record, config)
                total_count += 1
                if item["accepted"]:
                    accepted_count += 1
                    accepted_file.write(json.dumps(item, ensure_ascii=False))
                    accepted_file.write("\n")
                else:
                    rejected_count += 1
                    for reason in item["reject_reasons"]:
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
                    rejected_file.write(json.dumps(item, ensure_ascii=False))
                    rejected_file.write("\n")

                if total_count == 1 or total_count % 1000 == 0:
                    logger.info("Filtered transcript rows: %d", total_count)

            accepted_file.flush()
            rejected_file.flush()
            os.fsync(accepted_file.fileno())
            os.fsync(rejected_file.fileno())

        accepted_tmp.replace(accepted_path)
        rejected_tmp.replace(rejected_path)
        report = {
            "total_rows": total_count,
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "reject_reason_counts": dict(sorted(reason_counts.items())),
            "config": {
                "min_sec": config.filter.min_sec,
                "max_sec": config.filter.max_sec,
                "min_words": config.filter.min_words,
                "max_chars": config.filter.max_chars,
                "min_rms": config.filter.min_rms,
                "max_no_speech_prob": config.filter.max_no_speech_prob,
                "min_avg_logprob": config.filter.min_avg_logprob,
                "max_unusual_symbol_ratio": config.filter.max_unusual_symbol_ratio,
            },
        }
        save_json(report, report_path)
    except FilterError:
        raise
    except Exception as exc:
        raise FilterError(f"Failed to filter cleaned transcripts {input_path}: {exc}") from exc

    logger.info("Loaded %d cleaned transcript rows", total_count)
    logger.info("Accepted %d rows; rejected %d rows", accepted_count, rejected_count)
    logger.info("Saved accepted output: %s", accepted_path)
    logger.info("Saved rejected output: %s", rejected_path)
    logger.info("Saved filter report: %s", report_path)

    return {
        "total_rows": total_count,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
    }


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _compute_rms(path: Path) -> float:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise FilterError(
            "soundfile is required for RMS filtering. Run 'pip install -r requirements.txt'."
        ) from exc

    total_squares = 0.0
    total_samples = 0
    with sf.SoundFile(path, mode="r") as audio_file:
        while True:
            block = audio_file.read(frames=65536, dtype="float32", always_2d=True)
            if len(block) == 0:
                break
            total_squares += float((block * block).sum())
            total_samples += int(block.size)

    if total_samples == 0:
        return 0.0
    return math.sqrt(total_squares / total_samples)


def _unusual_symbol_ratio(text: str) -> float:
    unusual_count = sum(1 for char in text if not _COMMON_TEXT_RE.match(char))
    return unusual_count / max(1, len(text))
