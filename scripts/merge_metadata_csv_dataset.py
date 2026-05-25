"""Merge edited metadata.csv files into one minimal TTS dataset."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


class MetadataMergeError(RuntimeError):
    """Raised when metadata-only dataset merge cannot complete."""


@dataclass(frozen=True)
class MetadataRow:
    audio_name: str
    text: str


@dataclass(frozen=True)
class MergeStats:
    source_count: int
    row_count: int
    wav_count: int
    output_dir: Path
    metadata_path: Path
    wavs_dir: Path


def parse_metadata_line(line: str, source_path: Path, line_number: int) -> MetadataRow:
    """Parse one `filename.wav|text` metadata line."""

    stripped = line.rstrip("\n\r")
    if not stripped:
        raise MetadataMergeError(f"Empty metadata row at {source_path}:{line_number}")
    if "|" not in stripped:
        raise MetadataMergeError(
            f"Metadata row must contain '|': {source_path}:{line_number}"
        )

    audio_name, text = stripped.split("|", 1)
    audio_name = audio_name.strip()
    if not audio_name:
        raise MetadataMergeError(f"Metadata row has empty audio filename: {source_path}:{line_number}")
    if Path(audio_name).name != audio_name:
        raise MetadataMergeError(
            f"Metadata audio filename must not contain a path: {source_path}:{line_number}"
        )
    if text == "":
        raise MetadataMergeError(f"Metadata row has empty text: {source_path}:{line_number}")

    return MetadataRow(audio_name=audio_name, text=text)


def merge_metadata_dataset(
    source_root: str | Path,
    start: int,
    end: int,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> MergeStats:
    """Merge numbered mc*_stt metadata files and referenced WAVs into one folder."""

    root = Path(source_root)
    target_dir = Path(output_dir)
    if start < 1 or end < start:
        raise MetadataMergeError(f"Invalid source range: start={start}, end={end}")
    if not root.exists() or not root.is_dir():
        raise MetadataMergeError(f"Source root does not exist or is not a folder: {root}")

    source_dirs = [root / f"mc{index}_stt" for index in range(start, end + 1)]
    _validate_sources(source_dirs)
    _prepare_output_dir(target_dir, source_dirs, force=force)

    wavs_dir = target_dir / "wavs"
    metadata_path = target_dir / "metadata.csv"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    used_audio_names: set[str] = set()
    row_count = 0
    temp_metadata_path = metadata_path.with_name(f"{metadata_path.name}.tmp")

    try:
        with temp_metadata_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for source_dir in source_dirs:
                source_label = _safe_prefix(source_dir.name)
                source_metadata = source_dir / "metadata.csv"
                source_wavs = source_dir / "wavs"

                with source_metadata.open("r", encoding="utf-8-sig") as input_file:
                    for line_number, line in enumerate(input_file, start=1):
                        if not line.strip():
                            continue
                        row = parse_metadata_line(line, source_metadata, line_number)
                        source_audio = source_wavs / row.audio_name
                        if not source_audio.exists() or not source_audio.is_file():
                            raise MetadataMergeError(
                                "Metadata references missing WAV: "
                                f"{source_metadata}:{line_number} -> {source_audio}"
                            )

                        output_audio_name = _unique_audio_name(
                            f"{source_label}_{row.audio_name}",
                            used_audio_names,
                        )
                        shutil.copy2(source_audio, wavs_dir / output_audio_name)
                        output_file.write(f"{output_audio_name}|{row.text}\n")
                        row_count += 1

        temp_metadata_path.replace(metadata_path)
    except Exception:
        if temp_metadata_path.exists():
            temp_metadata_path.unlink()
        raise

    wav_count = len([path for path in wavs_dir.iterdir() if path.is_file()])
    if row_count == 0:
        raise MetadataMergeError("No metadata rows were merged.")
    if wav_count != row_count:
        raise MetadataMergeError(
            f"Merged WAV count does not match metadata rows: {wav_count} != {row_count}"
        )

    return MergeStats(
        source_count=len(source_dirs),
        row_count=row_count,
        wav_count=wav_count,
        output_dir=target_dir,
        metadata_path=metadata_path,
        wavs_dir=wavs_dir,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge edited metadata.csv files and referenced WAVs into one dataset."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/output"),
        help="Root folder containing mc*_stt output folders.",
    )
    parser.add_argument("--start", type=int, default=1, help="First numbered mc folder.")
    parser.add_argument("--end", type=int, default=31, help="Last numbered mc folder.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/datasets/mc1_mc31_metadata"),
        help="Output dataset folder.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the output folder first if it already exists.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        stats = merge_metadata_dataset(
            source_root=args.source_root,
            start=args.start,
            end=args.end,
            output_dir=args.output,
            force=args.force,
        )
    except MetadataMergeError as exc:
        parser.exit(status=1, message=f"metadata merge failed: {exc}\n")

    print(f"output_dir: {stats.output_dir}")
    print(f"wavs_dir: {stats.wavs_dir}")
    print(f"metadata_csv: {stats.metadata_path}")
    print(f"source_count: {stats.source_count}")
    print(f"metadata_rows: {stats.row_count}")
    print(f"wav_count: {stats.wav_count}")
    return 0


def _validate_sources(source_dirs: list[Path]) -> None:
    for source_dir in source_dirs:
        if not source_dir.exists() or not source_dir.is_dir():
            raise MetadataMergeError(f"Source folder does not exist: {source_dir}")
        metadata_path = source_dir / "metadata.csv"
        if not metadata_path.exists() or not metadata_path.is_file():
            raise MetadataMergeError(f"Source folder is missing metadata.csv: {source_dir}")
        wavs_dir = source_dir / "wavs"
        if not wavs_dir.exists() or not wavs_dir.is_dir():
            raise MetadataMergeError(f"Source folder is missing wavs/: {source_dir}")


def _prepare_output_dir(target_dir: Path, source_dirs: list[Path], *, force: bool) -> None:
    target_resolved = target_dir.resolve()
    for source_dir in source_dirs:
        source_resolved = source_dir.resolve()
        if target_resolved == source_resolved or _is_relative_to(target_resolved, source_resolved):
            raise MetadataMergeError(
                "Output folder must not be the same as or inside a source folder: "
                f"{target_dir}"
            )

    if target_dir.exists():
        if not force:
            raise MetadataMergeError(
                f"Output folder already exists: {target_dir}. Use --force to replace it."
            )
        if not target_dir.is_dir():
            raise MetadataMergeError(f"Output path exists and is not a folder: {target_dir}")
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=False)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _safe_prefix(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value.strip())
    safe = safe.strip("._-")
    return safe or "source"


def _unique_audio_name(candidate_name: str, used_names: set[str]) -> str:
    candidate_path = Path(candidate_name)
    stem = candidate_path.stem
    suffix = candidate_path.suffix
    unique_name = candidate_name
    index = 2
    while unique_name in used_names:
        unique_name = f"{stem}_{index}{suffix}"
        index += 1
    used_names.add(unique_name)
    return unique_name


if __name__ == "__main__":
    raise SystemExit(main())
