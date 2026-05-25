"""Transcript-first dataset workflow using ElevenLabs word timestamps."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.transcribe import load_elevenlabs_client
from src.utils import ensure_dir, save_json


logger = logging.getLogger("vn-audio2dataset.stt_workflow")

_SENTENCE_END_RE = re.compile(r"[.!?\u2026]+[\"')\]]*$")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_AUDIO_EVENT_RE = re.compile(r"^\s*[\[(<].+[\])>]\s*$")
_TRAILING_PUNCT_RE = re.compile(r"[^\wÀ-ỹ]+$", re.UNICODE)

_ENTITY_PREFIXES = {
    "anh",
    "bà",
    "ban",
    "bộ",
    "bộ trưởng",
    "công ty",
    "cộng hòa",
    "đảng",
    "đội",
    "đội tuyển",
    "giáo sư",
    "huyện",
    "ông",
    "phường",
    "quận",
    "sở",
    "tập đoàn",
    "thành phố",
    "thị xã",
    "thủ tướng",
    "tiến sĩ",
    "tỉnh",
    "trường",
    "xã",
}
_TRAILING_CONNECTORS = {
    "ai",
    "anh",
    "bằng",
    "bị",
    "các",
    "cái",
    "cho",
    "của",
    "đã",
    "đang",
    "để",
    "đến",
    "được",
    "gồm",
    "hay",
    "khi",
    "là",
    "mà",
    "một",
    "nên",
    "những",
    "ở",
    "sẽ",
    "tại",
    "theo",
    "thì",
    "trong",
    "và",
    "về",
    "với",
}
_LEADING_CONNECTORS = {
    "bởi",
    "cho",
    "của",
    "để",
    "hoặc",
    "là",
    "mà",
    "nên",
    "nhưng",
    "rằng",
    "thì",
    "và",
    "vì",
    "với",
}


class SttWorkflowError(RuntimeError):
    """Raised when transcript-first segmentation cannot complete."""


def transcribe_full_audio_with_elevenlabs(
    audio_path: str | Path,
    config: AppConfig,
) -> Any:
    """Transcribe a full preprocessed audio file with ElevenLabs Scribe."""

    path = Path(audio_path)
    if not path.exists():
        raise SttWorkflowError(f"STT input audio does not exist: {path}")
    if not path.is_file():
        raise SttWorkflowError(f"STT input audio path is not a file: {path}")

    client = load_elevenlabs_client()
    params: dict[str, Any] = {
        "model_id": config.stt_segmentation.model_id,
        "language_code": config.transcription.language,
        "timestamps_granularity": config.stt_segmentation.timestamps_granularity,
        "diarize": config.stt_segmentation.diarize,
        "tag_audio_events": config.stt_segmentation.tag_audio_events,
    }
    if config.stt_segmentation.num_speakers is not None:
        params["num_speakers"] = config.stt_segmentation.num_speakers
    if config.stt_segmentation.diarization_threshold is not None:
        params["diarization_threshold"] = config.stt_segmentation.diarization_threshold

    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            logger.info("Calling ElevenLabs full-audio STT: %s", path)
            with path.open("rb") as audio_file:
                return client.speech_to_text.convert(file=audio_file, **params)
        except Exception as exc:
            if attempt == attempts:
                raise SttWorkflowError(
                    f"ElevenLabs full-audio STT failed for {path}: {exc}"
                ) from exc
            delay_seconds = float(2 ** (attempt - 1))
            logger.warning(
                "ElevenLabs full-audio STT attempt %d/%d failed; retrying in %.1fs: %s",
                attempt,
                attempts,
                delay_seconds,
                exc,
            )
            time.sleep(delay_seconds)

    raise SttWorkflowError(f"ElevenLabs full-audio STT failed for {path}")


def save_full_stt_response(response: Any, output_path: str | Path) -> Path:
    """Save the full ElevenLabs response for audit/debugging."""

    return save_json(_json_safe(response), output_path)


def build_timestamp_segments(
    response: Any,
    config: AppConfig,
    input_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build conservative segment candidates from timestamped STT words."""

    words, invalid_word_count = normalize_stt_words(response)
    language = str(_response_value(response, "language_code") or config.transcription.language)
    full_text = str(_response_value(response, "text") or "").strip()
    target_speaker, speaker_stats, selection_info = select_target_speaker(
        words,
        config,
        input_path=input_path,
    )

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []

    for region in _dominant_speaker_regions(words, target_speaker, config):
        region_result = _segment_semantic_region(
            region,
            all_tokens=words,
            dominant_speaker=target_speaker,
            language=language,
            config=config,
        )
        for candidate in region_result["segments"]:
            candidate["id"] = f"{len(accepted) + 1:06d}"
            accepted.append(candidate)
        rejected.extend(region_result["rejected_segments"])
        boundary_rows.extend(region_result["boundary_scores"])

    report = {
        "language": language,
        "full_text_chars": len(full_text),
        "total_timestamped_tokens": len(words),
        "invalid_word_count": invalid_word_count,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted_duration_seconds": round(
            sum(float(item["duration"]) for item in accepted),
            3,
        ),
        "dominant_speaker": target_speaker,
        "selected_speaker": target_speaker,
        "speaker_selection": selection_info,
        "speaker_stats": speaker_stats,
        "config": {
            "model_id": config.stt_segmentation.model_id,
            "timestamps_granularity": config.stt_segmentation.timestamps_granularity,
            "diarize": config.stt_segmentation.diarize,
            "tag_audio_events": config.stt_segmentation.tag_audio_events,
            "min_speaker_share": config.stt_segmentation.min_speaker_share,
            "boundary_pad_sec": config.stt_segmentation.boundary_pad_sec,
            "boundary_guard_sec": config.stt_segmentation.boundary_guard_sec,
            "max_word_gap_sec": config.stt_segmentation.max_word_gap_sec,
            "preferred_min_sec": config.stt_segmentation.preferred_min_sec,
            "preferred_max_sec": config.stt_segmentation.preferred_max_sec,
            "semantic_max_sec": config.stt_segmentation.semantic_max_sec,
            "min_boundary_score": config.stt_segmentation.min_boundary_score,
            "allow_short_clips": config.stt_segmentation.allow_short_clips,
            "protect_named_entities": config.stt_segmentation.protect_named_entities,
            "protect_connector_phrases": config.stt_segmentation.protect_connector_phrases,
            "min_words": config.stt_segmentation.min_words,
            "min_avg_logprob": config.stt_segmentation.min_avg_logprob,
        },
        "reject_reason_counts": _reason_counts(rejected),
        "boundary_score_count": len(boundary_rows),
    }
    return {
        "words": words,
        "segments": accepted,
        "rejected_segments": rejected,
        "boundary_scores": boundary_rows,
        "report": report,
    }


def normalize_stt_words(response: Any) -> tuple[list[dict[str, Any]], int]:
    """Return timestamped token dictionaries from an ElevenLabs response."""

    raw_words = _response_value(response, "words") or []
    if not isinstance(raw_words, list):
        raise SttWorkflowError("ElevenLabs response did not contain a words list.")

    normalized: list[dict[str, Any]] = []
    invalid_count = 0
    for index, item in enumerate(raw_words):
        text = _token_text(item)
        start = _safe_float(_response_value(item, "start"))
        end = _safe_float(_response_value(item, "end"))
        if text == "" or start is None or end is None or end <= start:
            invalid_count += 1
            continue

        token_type = str(_response_value(item, "type") or "word").strip().lower()
        speaker_value = _response_value(item, "speaker_id")
        speaker_id = None if speaker_value in (None, "") else str(speaker_value)
        logprob = _safe_float(_response_value(item, "logprob"))
        normalized.append(
            {
                "index": index,
                "text": text,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "type": token_type,
                "speaker_id": speaker_id,
                "logprob": logprob,
            }
        )

    normalized.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    return normalized, invalid_count


def select_dominant_speaker(
    words: list[dict[str, Any]],
    config: AppConfig,
) -> tuple[str, dict[str, Any]]:
    """Select the main speaker by timestamped speech duration."""

    selected, stats, _selection = select_target_speaker(words, config)
    return selected, stats


def select_target_speaker(
    words: list[dict[str, Any]],
    config: AppConfig,
    input_path: str | Path | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Resolve the target speaker from config or auto speaker statistics."""

    stats = _speaker_statistics(words, config)
    speaker_stats = stats["speakers"]
    mode = _speaker_selection_mode(config.stt_segmentation.speaker_selection_mode)
    configured_target, selection_source = _configured_target_speaker(input_path, config)
    warnings: list[str] = []

    if not config.stt_segmentation.diarize:
        speech_duration = sum(
            max(0.0, float(item["end"]) - float(item["start"]))
            for item in words
            if _is_speech_token(item)
        )
        selected = "speaker_0"
        return selected, {
            "speaker_0": {
                "duration_seconds": round(speech_duration, 3),
                "share": 1.0,
                "word_count": sum(1 for item in words if _is_speech_token(item)),
            }
        }, {
            "selected_speaker": selected,
            "selection_mode": mode,
            "selection_source": "diarization_disabled",
            "configured_target": configured_target,
            "selection_warnings": [],
        }

    if not speaker_stats:
        raise SttWorkflowError(
            "No diarized speech words were found in the ElevenLabs response."
        )

    if configured_target != "auto":
        if configured_target not in speaker_stats:
            raise SttWorkflowError(
                f"Configured target speaker '{configured_target}' was not found."
            )
        selected = configured_target
        selection_mode = "manual" if mode == "manual" else mode
    elif mode == "manual" or config.stt_segmentation.require_manual_speaker:
        raise SttWorkflowError(
            "Manual speaker selection is required, but no target speaker was "
            f"configured for {input_path or '<input>'}."
        )
    else:
        selected = _auto_select_speaker(speaker_stats)
        selection_source = "auto"
        selection_mode = mode

    selected_share = float(speaker_stats[selected].get("share", 0.0))
    if selected_share < config.stt_segmentation.min_speaker_share:
        warnings.append("low_speaker_share")
    if _has_close_second_speaker(speaker_stats, selected):
        warnings.append("close_second_speaker")
    if int(stats.get("missing_speaker_word_count", 0)) > 0:
        warnings.append("missing_speaker_words")

    selection_info = {
        "selected_speaker": selected,
        "selection_mode": selection_mode,
        "selection_source": selection_source,
        "configured_target": configured_target,
        "selection_warnings": sorted(set(warnings)),
    }
    return selected, speaker_stats, selection_info


def build_speaker_inspection(response: Any, config: AppConfig) -> dict[str, Any]:
    """Build diarization diagnostics without creating dataset clips."""

    words, invalid_word_count = normalize_stt_words(response)
    stats = _speaker_statistics(words, config)
    return {
        "words": words,
        "speaker_inspection": {
            "total_timestamped_tokens": len(words),
            "invalid_word_count": invalid_word_count,
            "speaker_count": len(stats["speakers"]),
            "missing_speaker_word_count": stats["missing_speaker_word_count"],
            "audio_event_count": stats["audio_event_count"],
            "speakers": stats["speakers"],
        },
        "speaker_turns": stats["turns"],
    }


def _speaker_statistics(words: list[dict[str, Any]], config: AppConfig) -> dict[str, Any]:
    turns = _speaker_turns(words, config)
    speaker_totals: dict[str, dict[str, Any]] = {}
    missing_speaker_words = 0
    audio_event_count = 0

    for item in words:
        if _is_audio_event_token(item):
            audio_event_count += 1
            continue
        if not _is_speech_token(item):
            continue
        speaker_id = _speaker_for_token(item, config)
        if speaker_id in (None, ""):
            missing_speaker_words += 1
            continue
        stats = speaker_totals.setdefault(
            str(speaker_id),
            {
                "duration_seconds": 0.0,
                "word_count": 0,
                "first_start": None,
                "last_end": None,
            },
        )
        start = float(item["start"])
        end = float(item["end"])
        stats["duration_seconds"] += max(0.0, end - start)
        stats["word_count"] += 1
        stats["first_start"] = start if stats["first_start"] is None else min(stats["first_start"], start)
        stats["last_end"] = end if stats["last_end"] is None else max(stats["last_end"], end)

    total_duration = sum(float(item["duration_seconds"]) for item in speaker_totals.values())
    by_speaker_turns: dict[str, list[dict[str, Any]]] = {}
    for turn in turns:
        by_speaker_turns.setdefault(str(turn["speaker_id"]), []).append(turn)

    speakers: dict[str, Any] = {}
    for speaker_id, raw_stats in sorted(speaker_totals.items()):
        speaker_turns = by_speaker_turns.get(speaker_id, [])
        turn_durations = [float(item["duration"]) for item in speaker_turns]
        speakers[speaker_id] = {
            "duration_seconds": round(float(raw_stats["duration_seconds"]), 3),
            "share": round(float(raw_stats["duration_seconds"]) / max(total_duration, 1e-9), 4),
            "word_count": int(raw_stats["word_count"]),
            "turn_count": len(speaker_turns),
            "average_turn_duration_seconds": round(
                sum(turn_durations) / len(turn_durations), 3
            )
            if turn_durations
            else 0.0,
            "first_start": round(float(raw_stats["first_start"]), 3)
            if raw_stats["first_start"] is not None
            else None,
            "last_end": round(float(raw_stats["last_end"]), 3)
            if raw_stats["last_end"] is not None
            else None,
            "sample_utterances": [
                str(item["text"])
                for item in speaker_turns[:3]
                if str(item.get("text", "")).strip()
            ],
        }

    return {
        "speakers": speakers,
        "turns": turns,
        "missing_speaker_word_count": missing_speaker_words,
        "audio_event_count": audio_event_count,
    }


def _speaker_turns(words: list[dict[str, Any]], config: AppConfig) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_speaker: str | None = None

    def flush() -> None:
        nonlocal current, current_speaker
        if not current or current_speaker is None:
            current = []
            current_speaker = None
            return
        start = float(current[0]["start"])
        end = float(current[-1]["end"])
        turns.append(
            {
                "speaker_id": current_speaker,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "word_count": _speech_word_count(current),
                "text": _join_text(current),
                "word_indexes": [int(item["index"]) for item in current],
            }
        )
        current = []
        current_speaker = None

    for token in words:
        if not _is_speech_token(token):
            flush()
            continue
        speaker = _speaker_for_token(token, config)
        if speaker is None:
            flush()
            continue
        if current and speaker != current_speaker:
            flush()
        if current:
            gap = max(0.0, float(token["start"]) - float(current[-1]["end"]))
            if gap > config.stt_segmentation.max_word_gap_sec:
                flush()
        current.append(token)
        current_speaker = speaker

    flush()
    return turns


def _speaker_selection_mode(value: str) -> str:
    mode = str(value or "auto").strip().lower().replace("-", "_")
    if mode not in {"auto", "auto_with_warning", "manual"}:
        raise SttWorkflowError(
            "stt_segmentation.speaker_selection_mode must be one of "
            "auto, auto_with_warning, or manual."
        )
    return mode


def _configured_target_speaker(
    input_path: str | Path | None,
    config: AppConfig,
) -> tuple[str, str]:
    mapping = config.stt_segmentation.per_file_target_speakers
    if input_path is not None:
        for key in _input_mapping_keys(Path(input_path)):
            if key in mapping:
                return str(mapping[key]).strip(), "per_file"

    default_target = str(config.stt_segmentation.default_target_speaker or "auto").strip()
    if default_target and default_target.lower() != "auto":
        return default_target, "default"
    legacy_target = str(config.stt_segmentation.dominant_speaker or "auto").strip()
    if legacy_target and legacy_target.lower() != "auto":
        return legacy_target, "legacy_dominant_speaker"
    return "auto", "auto"


def _input_mapping_keys(input_path: Path) -> list[str]:
    keys = [
        str(input_path),
        input_path.as_posix(),
        input_path.name,
        input_path.stem,
        os.path.normpath(str(input_path)),
    ]
    try:
        relative = input_path.resolve().relative_to(Path.cwd().resolve())
        keys.extend([str(relative), relative.as_posix()])
    except Exception:
        pass
    seen: set[str] = set()
    return [key for key in keys if key and not (key in seen or seen.add(key))]


def _auto_select_speaker(speaker_stats: dict[str, Any]) -> str:
    return max(
        speaker_stats,
        key=lambda speaker_id: (
            float(speaker_stats[speaker_id].get("duration_seconds", 0.0)),
            int(speaker_stats[speaker_id].get("word_count", 0)),
        ),
    )


def _has_close_second_speaker(
    speaker_stats: dict[str, Any],
    selected_speaker: str,
) -> bool:
    selected_share = float(speaker_stats[selected_speaker].get("share", 0.0))
    other_shares = [
        float(stats.get("share", 0.0))
        for speaker_id, stats in speaker_stats.items()
        if speaker_id != selected_speaker
    ]
    return bool(other_shares) and selected_share - max(other_shares) <= 0.10


def _dominant_speaker_regions(
    words: list[dict[str, Any]],
    dominant_speaker: str,
    config: AppConfig,
) -> list[list[dict[str, Any]]]:
    regions: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal current
        if current:
            regions.append(current)
            current = []

    for token in words:
        if not _is_speech_token(token):
            flush()
            continue

        if _speaker_for_token(token, config) != dominant_speaker:
            flush()
            continue

        if current:
            previous = current[-1]
            gap = max(0.0, float(token["start"]) - float(previous["end"]))
            if gap > config.stt_segmentation.max_word_gap_sec:
                flush()

        current.append(token)

    flush()
    return regions


def _segment_semantic_region(
    region_words: list[dict[str, Any]],
    all_tokens: list[dict[str, Any]],
    dominant_speaker: str,
    language: str,
    config: AppConfig,
) -> dict[str, list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    boundary_scores: list[dict[str, Any]] = []
    start_index = 0

    while start_index < len(region_words):
        selection = _select_semantic_boundary(region_words, start_index, config)
        boundary_scores.extend(selection["boundary_scores"])
        end_index = selection.get("end_index")

        if end_index is None:
            rejected.append(
                _rejected_region_candidate(
                    region_words[start_index:],
                    dominant_speaker,
                    language,
                    config,
                    ["no_safe_semantic_boundary"],
                )
            )
            break

        candidate = _candidate_from_words(
            region_words[start_index : int(end_index) + 1],
            dominant_speaker,
            language,
            config,
            source="stt_semantic",
        )
        candidate["boundary_score"] = selection["score"]
        candidate["boundary_reasons"] = selection["reasons"]
        validation_reasons = validate_candidate(candidate, all_tokens, dominant_speaker, config)
        if validation_reasons:
            rejected.append({**candidate, "reject_reasons": validation_reasons})
        else:
            accepted.append(candidate)

        start_index = int(end_index) + 1

    return {
        "segments": accepted,
        "rejected_segments": rejected,
        "boundary_scores": boundary_scores,
    }


def _select_semantic_boundary(
    region_words: list[dict[str, Any]],
    start_index: int,
    config: AppConfig,
) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for end_index in range(start_index, len(region_words)):
        score = _score_boundary(region_words, start_index, end_index, config)
        scored.append(score)
        if (
            float(score["duration"]) > _stt_max_sec(config)
            and "sentence_punctuation" in score["reasons"]
        ):
            break

    valid = [
        item
        for item in scored
        if item["is_valid"]
        and float(item["score"]) >= config.stt_segmentation.min_boundary_score
    ]
    if not valid:
        return {"end_index": None, "boundary_scores": scored}

    preferred = [
        item
        for item in valid
        if config.stt_segmentation.preferred_min_sec
        <= float(item["duration"])
        <= config.stt_segmentation.preferred_max_sec
    ]
    extended = [
        item
        for item in valid
        if config.stt_segmentation.preferred_max_sec
        < float(item["duration"])
        <= _stt_max_sec(config)
    ]
    overlong_sentence = [
        item
        for item in valid
        if float(item["duration"]) > _stt_max_sec(config)
        and "sentence_punctuation" in item["reasons"]
    ]
    short = [
        item
        for item in valid
        if _stt_min_sec(config)
        <= float(item["duration"])
        < config.stt_segmentation.preferred_min_sec
    ]

    for bucket in (preferred, extended, overlong_sentence, short):
        if bucket:
            best = max(bucket, key=lambda item: (float(item["score"]), float(item["duration"])))
            return {**best, "boundary_scores": scored}

    return {"end_index": None, "boundary_scores": scored}


def _score_boundary(
    region_words: list[dict[str, Any]],
    start_index: int,
    end_index: int,
    config: AppConfig,
) -> dict[str, Any]:
    start_word = region_words[start_index]
    left = region_words[end_index]
    right = region_words[end_index + 1] if end_index + 1 < len(region_words) else None
    duration = float(left["end"]) - float(start_word["start"])
    reasons: list[str] = []
    protections = _boundary_protection_reasons(region_words, end_index, config)

    score = 0.0
    sentence_final = _looks_sentence_final(str(left["text"]))
    if sentence_final:
        score += config.stt_segmentation.sentence_punctuation_weight
        reasons.append("sentence_punctuation")

    gap = None
    if right is None:
        score += config.stt_segmentation.sentence_punctuation_weight
        reasons.append("region_end")
    else:
        gap = max(0.0, float(right["start"]) - float(left["end"]))
        if gap >= config.stt_segmentation.pause_strong_sec:
            score += config.stt_segmentation.pause_strong_weight
            reasons.append("strong_pause")
        elif gap >= config.stt_segmentation.pause_medium_sec:
            score += config.stt_segmentation.pause_medium_weight
            reasons.append("medium_pause")

    if config.stt_segmentation.preferred_min_sec <= duration <= config.stt_segmentation.preferred_max_sec:
        score += 3.0
        reasons.append("preferred_duration")
        midpoint = (
            config.stt_segmentation.preferred_min_sec
            + config.stt_segmentation.preferred_max_sec
        ) / 2.0
        score += max(0.0, 2.0 - abs(duration - midpoint) * 0.35)
    elif config.stt_segmentation.preferred_max_sec < duration <= _stt_max_sec(config):
        score += 1.0
        reasons.append("extended_duration")
    elif _stt_min_sec(config) <= duration < config.stt_segmentation.preferred_min_sec:
        score -= 1.0
        reasons.append("short_duration")

    if right is not None and not sentence_final:
        gap_value = 0.0 if gap is None else gap
        if gap_value < config.stt_segmentation.pause_medium_sec:
            score -= 4.0
            reasons.append("no_punctuation_or_pause")
        if _is_lowercase_word(str(right["text"])):
            score -= 2.0
            reasons.append("lowercase_continuation")

    if protections:
        reasons.extend(protections)

    is_valid = True
    if duration < _stt_min_sec(config) and not config.stt_segmentation.allow_short_clips:
        is_valid = False
        reasons.append("duration_too_short")
    if duration > _stt_max_sec(config) and not sentence_final:
        is_valid = False
        reasons.append("duration_too_long")
    if right is not None and not sentence_final:
        is_valid = False
        reasons.append("not_sentence_boundary")
    if protections:
        is_valid = False

    return {
        "start_index": start_index,
        "end_index": end_index,
        "left_text": str(left["text"]),
        "right_text": str(right["text"]) if right is not None else None,
        "duration": round(duration, 3),
        "gap_after": round(gap, 3) if gap is not None else None,
        "score": round(score, 3),
        "is_valid": is_valid,
        "reasons": sorted(set(reasons)),
    }


def _boundary_protection_reasons(
    words: list[dict[str, Any]],
    end_index: int,
    config: AppConfig,
) -> list[str]:
    if end_index + 1 >= len(words):
        return []

    left = _clean_word(str(words[end_index]["text"]))
    right = _clean_word(str(words[end_index + 1]["text"]))
    left_lower = left.lower()
    right_lower = right.lower()
    reasons: list[str] = []

    if config.stt_segmentation.protect_named_entities:
        if _starts_upper(left) and _starts_upper(right):
            reasons.append("protected_capitalized_entity")
        for size in (3, 2, 1):
            prefix = _lower_phrase(words, max(0, end_index - size + 1), end_index)
            if prefix in _ENTITY_PREFIXES and _starts_upper(right):
                reasons.append("protected_entity_prefix")
                break
        if left_lower in {"đông", "tây", "nam", "bắc", "việt"} and _starts_upper(right):
            reasons.append("protected_direction_or_country_name")

    if config.stt_segmentation.protect_connector_phrases:
        if left_lower in _TRAILING_CONNECTORS:
            reasons.append("protected_trailing_connector")
        if right_lower in _LEADING_CONNECTORS and not _looks_sentence_final(str(words[end_index]["text"])):
            reasons.append("protected_leading_connector")

    return sorted(set(reasons))


def _rejected_region_candidate(
    words: list[dict[str, Any]],
    dominant_speaker: str,
    language: str,
    config: AppConfig,
    reasons: list[str],
) -> dict[str, Any]:
    candidate = _candidate_from_words(
        words,
        dominant_speaker,
        language,
        config,
        source="stt_rejected_region",
    )
    candidate["reject_reasons"] = sorted(set(reasons))
    return candidate


def validate_candidate(
    candidate: dict[str, Any],
    all_tokens: list[dict[str, Any]],
    dominant_speaker: str,
    config: AppConfig,
) -> list[str]:
    """Return reject reasons for a candidate, or an empty list when safe."""

    reasons: list[str] = []
    start = float(candidate["start"])
    end = float(candidate["end"])
    raw_start = float(candidate["raw_start"])
    raw_end = float(candidate["raw_end"])
    duration = end - start

    if duration < _stt_min_sec(config) and not config.stt_segmentation.allow_short_clips:
        reasons.append("duration_too_short")
    if duration > _stt_max_sec(config) and "sentence_punctuation" not in candidate.get(
        "boundary_reasons",
        [],
    ):
        reasons.append("duration_too_long")
    if int(candidate["word_count"]) < config.stt_segmentation.min_words:
        reasons.append("word_count_too_low")
    if str(candidate.get("text", "")).strip() == "":
        reasons.append("empty_text")

    avg_logprob = _safe_float(candidate.get("avg_logprob"))
    if (
        config.stt_segmentation.min_avg_logprob is not None
        and avg_logprob is not None
        and avg_logprob < config.stt_segmentation.min_avg_logprob
    ):
        reasons.append("avg_logprob_too_low")

    for token in _overlapping_tokens(all_tokens, raw_start, raw_end):
        if _is_audio_event_token(token):
            reasons.append("audio_event_overlap")
            break
    for token in _overlapping_tokens(all_tokens, raw_start, raw_end):
        if not _is_speech_token(token):
            continue
        speaker = _speaker_for_token(token, config)
        if speaker is None:
            reasons.append("unknown_speaker_overlap")
            break
        if speaker != dominant_speaker:
            reasons.append("mixed_speaker_overlap")
            break

    previous_token = _previous_token(all_tokens, raw_start)
    if _is_risky_boundary_token(previous_token, dominant_speaker, config):
        gap = raw_start - float(previous_token["end"])
        if gap < config.stt_segmentation.boundary_guard_sec:
            reasons.append("risky_leading_transition")

    next_token = _next_token(all_tokens, raw_end)
    if _is_risky_boundary_token(next_token, dominant_speaker, config):
        gap = float(next_token["start"]) - raw_end
        if gap < config.stt_segmentation.boundary_guard_sec:
            reasons.append("risky_trailing_transition")

    return sorted(set(reasons))


def write_transcripts_from_stt_segments(
    manifest_items: list[dict[str, Any]],
    stt_segments: list[dict[str, Any]],
    output_path: str | Path,
    config: AppConfig,
) -> list[dict[str, Any]]:
    """Write raw transcript JSONL by pairing quality-approved audio with STT text."""

    output = Path(output_path)
    temp_path = output.with_name(f"{output.name}.tmp")
    ensure_dir(output.parent)

    by_id = {str(item["id"]): item for item in stt_segments}
    records: list[dict[str, Any]] = []
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as file:
            for item in manifest_items:
                segment = by_id.get(str(item.get("id", "")))
                if segment is None:
                    raise SttWorkflowError(
                        f"Audio quality manifest references unknown STT segment {item.get('id')!r}."
                    )
                record = {
                    "id": str(item["id"]),
                    "audio_path": str(item["audio_path"]),
                    "text": str(segment["text"]),
                    "language": str(segment.get("language") or config.transcription.language),
                    "duration": float(item["duration"]),
                    "avg_logprob": segment.get("avg_logprob"),
                    "no_speech_prob": None,
                    "speaker_id": segment.get("speaker_id"),
                    "start": item.get("start", segment.get("start")),
                    "end": item.get("end", segment.get("end")),
                    "error": None,
                }
                records.append(record)
                file.write(json.dumps(_json_safe(record), ensure_ascii=False))
                file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temp_path.replace(output)
    except Exception as exc:
        raise SttWorkflowError(f"Failed to write STT raw transcripts to {output}: {exc}") from exc

    logger.info("Saved STT-derived raw transcripts: %s (%d rows)", output, len(records))
    return records


def _candidate_from_words(
    words: list[dict[str, Any]],
    dominant_speaker: str,
    language: str,
    config: AppConfig,
    source: str,
) -> dict[str, Any]:
    raw_start = float(words[0]["start"])
    raw_end = float(words[-1]["end"])
    start = max(0.0, raw_start - config.stt_segmentation.boundary_pad_sec)
    end = raw_end + config.stt_segmentation.boundary_pad_sec
    logprob_values = [
        float(item["logprob"])
        for item in words
        if isinstance(item.get("logprob"), (int, float))
        and math.isfinite(float(item["logprob"]))
    ]
    return {
        "id": "",
        "start": round(start, 3),
        "end": round(end, 3),
        "raw_start": round(raw_start, 3),
        "raw_end": round(raw_end, 3),
        "duration": round(end - start, 3),
        "text": _join_text(words),
        "language": language,
        "speaker_id": dominant_speaker,
        "word_count": _speech_word_count(words),
        "avg_logprob": round(sum(logprob_values) / len(logprob_values), 4)
        if logprob_values
        else None,
        "source": source,
        "word_indexes": [int(item["index"]) for item in words],
    }


def _join_text(words: list[dict[str, Any]]) -> str:
    text = " ".join(str(item["text"]).strip() for item in words if str(item["text"]).strip())
    text = re.sub(r"\s+([,.?!:;])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _stt_min_sec(config: AppConfig) -> float:
    return float(config.segments.min_sec)


def _stt_max_sec(config: AppConfig) -> float:
    return min(float(config.segments.max_sec), float(config.stt_segmentation.semantic_max_sec))


def _speech_word_count(words: list[dict[str, Any]]) -> int:
    return sum(1 for item in words if _WORD_RE.search(str(item.get("text", ""))))


def _is_speech_token(token: dict[str, Any]) -> bool:
    token_type = str(token.get("type") or "word").lower()
    if _is_audio_event_token(token):
        return False
    if token_type in {"spacing", "space"}:
        return False
    return str(token.get("text", "")).strip() != ""


def _is_audio_event_token(token: dict[str, Any]) -> bool:
    token_type = str(token.get("type") or "").lower()
    text = str(token.get("text", ""))
    return "event" in token_type or bool(_AUDIO_EVENT_RE.match(text))


def _speaker_for_token(token: dict[str, Any], config: AppConfig) -> str | None:
    if not config.stt_segmentation.diarize:
        return "speaker_0"
    value = token.get("speaker_id")
    if value in (None, ""):
        return None
    return str(value)


def _overlapping_tokens(
    tokens: list[dict[str, Any]],
    start: float,
    end: float,
) -> list[dict[str, Any]]:
    return [
        item
        for item in tokens
        if float(item["start"]) < end and float(item["end"]) > start
    ]


def _previous_token(tokens: list[dict[str, Any]], start: float) -> dict[str, Any] | None:
    previous = [item for item in tokens if float(item["end"]) <= start]
    if not previous:
        return None
    return max(previous, key=lambda item: float(item["end"]))


def _next_token(tokens: list[dict[str, Any]], end: float) -> dict[str, Any] | None:
    following = [item for item in tokens if float(item["start"]) >= end]
    if not following:
        return None
    return min(following, key=lambda item: float(item["start"]))


def _is_risky_boundary_token(
    token: dict[str, Any] | None,
    dominant_speaker: str,
    config: AppConfig,
) -> bool:
    if token is None:
        return False
    if _is_audio_event_token(token):
        return True
    if not _is_speech_token(token):
        return False
    return _speaker_for_token(token, config) != dominant_speaker


def _looks_sentence_final(text: str) -> bool:
    return bool(_SENTENCE_END_RE.search(text.strip()))


def _clean_word(text: str) -> str:
    return _TRAILING_PUNCT_RE.sub("", text.strip())


def _lower_phrase(words: list[dict[str, Any]], start_index: int, end_index: int) -> str:
    parts = [
        _clean_word(str(words[index]["text"])).lower()
        for index in range(start_index, end_index + 1)
    ]
    return " ".join(part for part in parts if part)


def _starts_upper(text: str) -> bool:
    value = _clean_word(text)
    return bool(value) and value[0].isupper()


def _is_lowercase_word(text: str) -> bool:
    value = _clean_word(text)
    return bool(value) and value[0].islower()


def _token_text(item: Any) -> str:
    value = _response_value(item, "text")
    if value in (None, ""):
        value = _response_value(item, "word")
    return str(value or "").strip()


def _response_value(response: Any, key: str) -> Any:
    if isinstance(response, dict):
        return response.get(key)
    if hasattr(response, key):
        return getattr(response, key)
    if hasattr(response, "model_dump"):
        try:
            value = response.model_dump()
        except Exception:
            return None
        if isinstance(value, dict):
            return value.get(key)
    return None


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _reason_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        reasons = item.get("reject_reasons", [])
        if not isinstance(reasons, list):
            continue
        for reason in reasons:
            reason_key = str(reason)
            counts[reason_key] = counts.get(reason_key, 0) + 1
    return dict(sorted(counts.items()))


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            value = value.model_dump()
        except Exception:
            value = str(value)
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
