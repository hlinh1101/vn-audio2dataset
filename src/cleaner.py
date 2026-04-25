"""Rule-based transcript cleaning for Vietnamese TTS data."""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.utils import ensure_dir


logger = logging.getLogger("vn-audio2dataset.cleaner")

_WHITESPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,?!:;])")
_SPACE_AFTER_PUNCT_RE = re.compile(r"([.,?!:;])(?=\S)")
_DUPLICATE_PUNCT_RE = re.compile(r"([.,?!:;])\1+")
_DUPLICATE_HYPHEN_RE = re.compile(r"-{2,}")
_QUOTE_CHARS = "\"\u201c\u201d\u2018\u2019`\u00b4"
_ALLOWED_PUNCTUATION = {".", ",", "?", "!", ":", ";", "-", "'"}
_ZERO_WIDTH_CATEGORIES = {"Cf"}


class CleanerError(RuntimeError):
    """Raised when transcript cleaning cannot complete."""


def clean_text(text: str, config: AppConfig) -> str:
    """Normalize one transcript string without changing its meaning."""

    value = str(text).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    value = unicodedata.normalize("NFC", value)
    value = "".join(_clean_char(char, config) for char in value)

    if config.cleaning.strip_quotes:
        value = value.strip(_QUOTE_CHARS)

    value = _DUPLICATE_PUNCT_RE.sub(r"\1", value)
    value = _DUPLICATE_HYPHEN_RE.sub("-", value)
    value = _WHITESPACE_RE.sub(" ", value)
    value = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", value)
    value = _SPACE_AFTER_PUNCT_RE.sub(r"\1 ", value)
    value = _WHITESPACE_RE.sub(" ", value).strip()

    if config.cleaning.lowercase_text:
        value = value.lower()

    return value


def clean_transcript_record(record: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    """Return a transcript record with cleaning metadata added."""

    raw_text = record.get("text", "")
    original_text = raw_text if isinstance(raw_text, str) else str(raw_text)
    cleaned_text = clean_text(original_text, config)
    cleaned = dict(record)
    cleaned["cleaned_text"] = cleaned_text
    cleaned["is_empty_after_cleaning"] = cleaned_text == ""
    cleaned["cleaning_changed_text"] = cleaned_text != original_text
    return cleaned


def clean_all_transcripts(
    input_path: str | Path,
    output_path: str | Path,
    config: AppConfig,
) -> dict[str, int]:
    """Clean all transcript JSONL records and write cleaned JSONL output."""

    source_path = Path(input_path)
    target_path = Path(output_path)
    temp_path = target_path.with_name(f"{target_path.name}.tmp")
    if not source_path.exists():
        raise CleanerError(f"Raw transcripts file does not exist: {source_path}")
    if not source_path.is_file():
        raise CleanerError(f"Raw transcripts path is not a file: {source_path}")

    ensure_dir(target_path.parent)
    total_rows = 0
    changed_rows = 0
    empty_rows = 0

    logger.info("Cleaning transcripts from %s", source_path)
    logger.info("Saving cleaned transcripts to %s via temp file %s", target_path, temp_path)
    try:
        with source_path.open("r", encoding="utf-8-sig") as input_file, temp_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as output_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CleanerError(
                        f"Invalid JSONL at {source_path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise CleanerError(
                        f"Transcript record at {source_path}:{line_number} must be an object."
                    )

                cleaned_record = clean_transcript_record(record, config)
                total_rows += 1
                if cleaned_record["cleaning_changed_text"]:
                    changed_rows += 1
                if cleaned_record["is_empty_after_cleaning"]:
                    empty_rows += 1

                output_file.write(json.dumps(cleaned_record, ensure_ascii=False))
                output_file.write("\n")

                if total_rows == 1 or total_rows % 1000 == 0:
                    logger.info("Cleaned transcript rows: %d", total_rows)

            output_file.flush()
            os.fsync(output_file.fileno())

        temp_path.replace(target_path)
        if not target_path.exists():
            raise CleanerError(f"Cleaned transcript output was not created: {target_path}")
        if total_rows > 0 and target_path.stat().st_size == 0:
            raise CleanerError(f"Cleaned transcript output is empty: {target_path}")
    except CleanerError:
        raise
    except Exception as exc:
        raise CleanerError(f"Failed to clean transcripts {source_path}: {exc}") from exc

    logger.info("Loaded %d transcript rows from %s", total_rows, source_path)
    logger.info(
        "Cleaning changed %d rows; %d rows are empty after cleaning",
        changed_rows,
        empty_rows,
    )
    logger.info(
        "Saved cleaned transcripts: %s (%d rows, %d changed, %d empty)",
        target_path,
        total_rows,
        changed_rows,
        empty_rows,
    )
    return {
        "total_rows": total_rows,
        "cleaned_rows": changed_rows,
        "empty_rows_after_cleaning": empty_rows,
    }


def _clean_char(char: str, config: AppConfig) -> str:
    if unicodedata.category(char) in _ZERO_WIDTH_CATEGORIES:
        return ""
    if config.cleaning.remove_emojis and _is_emoji(char):
        return ""
    if char.isalnum() or char.isspace():
        return char
    if char in _ALLOWED_PUNCTUATION or char in _QUOTE_CHARS:
        return char
    if unicodedata.category(char).startswith("M"):
        return char
    return " "


def _is_emoji(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x1F300 <= codepoint <= 0x1FAFF
        or 0x2600 <= codepoint <= 0x27BF
        or 0xFE00 <= codepoint <= 0xFE0F
    )
