"""Small reusable helpers for filesystem and serialization tasks."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def timestamp_str() -> str:
    """Return a compact timestamp suitable for filenames."""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_stem(path: str | Path) -> str:
    """Return a filesystem-friendly stem from a path-like value."""

    stem = Path(path).stem.strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    safe = safe.strip("._-")
    return safe or "audio"


def save_json(data: Any, path: str | Path) -> Path:
    """Write JSON data using UTF-8 and stable formatting."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return output_path


def load_json(path: str | Path) -> Any:
    """Read JSON data from a UTF-8 file."""

    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as file:
        return json.load(file)
