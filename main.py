"""CLI entrypoint for vn-audio2dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.cleaner import CleanerError, clean_all_transcripts
from src.config import AppConfig, load_config
from src.exporter import (
    ExporterError,
    cut_audio_segments,
    export_dataset_from_accepted,
    load_final_segments,
)
from src.filter import FilterError, filter_all
from src.logger import setup_logger
from src.preprocess import PreprocessError, preprocess_audio
from src.segmenter import (
    SegmenterError,
    build_segments,
    load_vad_segments,
    save_final_segments,
)
from src.transcribe import (
    TranscriptionError,
    transcribe_all_segments,
)
from src.utils import ensure_dir, safe_stem, save_json
from src.vad import VadError, run_vad, save_vad_segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess audio, run Silero VAD, build segment metadata, "
            "export segment WAV files, transcribe them, clean text, filter quality, "
            "and export final dataset metadata."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("full", "clean", "filter", "export"),
        default="full",
        help=(
            "Pipeline stage to run. Use 'clean' for existing raw transcripts or "
            "'filter' for existing cleaned transcripts or 'export' for accepted records."
        ),
    )
    parser.add_argument(
        "--input",
        required=False,
        type=Path,
        help="Path to the long input audio file.",
    )
    parser.add_argument(
        "--raw",
        required=False,
        type=Path,
        help="Path to raw_transcripts.jsonl for clean-only mode.",
    )
    parser.add_argument(
        "--cleaned",
        required=False,
        type=Path,
        help="Path to cleaned_transcripts.jsonl for filter-only mode.",
    )
    parser.add_argument(
        "--accepted",
        required=False,
        type=Path,
        help="Path to accepted.jsonl for export-only mode.",
    )
    parser.add_argument(
        "--output",
        required=False,
        type=Path,
        help="Processed working directory. Defaults to paths.processed_dir from config.",
    )
    parser.add_argument(
        "--config",
        default=Path("config.yaml"),
        type=Path,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    if args.stage == "clean" and args.raw is None:
        parser.error("--raw is required when --stage clean is used.")
    if args.stage == "filter" and args.cleaned is None:
        parser.error("--cleaned is required when --stage filter is used.")
    if args.stage == "export" and args.accepted is None:
        parser.error("--accepted is required when --stage export is used.")
    if args.stage == "full" and args.input is None:
        parser.error(
            "--input is required unless --stage clean, --stage filter, or --stage export is used."
        )
    return args


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger(
        name=config.project_name,
        level=config.logging.level,
        log_file=config.logging.log_file,
    )

    logger.info("Loaded config from %s", args.config)
    logger.info("Running mode: %s", args.stage)

    if args.stage == "clean":
        return run_clean_stage(args.raw, config)
    if args.stage == "filter":
        return run_filter_stage(args.cleaned, config)
    if args.stage == "export":
        return run_export_stage(args.accepted, config)

    return run_full_pipeline(args, config)


def run_clean_stage(raw_path: Path, config: AppConfig) -> int:
    logger = logging.getLogger(config.project_name)
    cleaned_transcripts_path = raw_path.parent / "cleaned_transcripts.jsonl"
    try:
        cleaning_stats = clean_all_transcripts(
            raw_path,
            cleaned_transcripts_path,
            config,
        )
    except CleanerError as exc:
        logger.exception("Clean stage failed: %s", exc)
        return 1

    print(f"raw_transcripts: {raw_path}")
    print(f"cleaned_transcripts: {cleaned_transcripts_path}")
    print(f"total_transcript_rows: {cleaning_stats['total_rows']}")
    print(f"cleaned_rows_count: {cleaning_stats['cleaned_rows']}")
    print(f"empty_rows_after_cleaning_count: {cleaning_stats['empty_rows_after_cleaning']}")
    return 0


def run_filter_stage(cleaned_path: Path, config: AppConfig) -> int:
    logger = logging.getLogger(config.project_name)
    output_dir = cleaned_path.parent
    accepted_path = output_dir / "accepted.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    report_path = output_dir / "filter_report.json"
    try:
        filter_stats = filter_all(
            cleaned_path,
            accepted_path,
            rejected_path,
            report_path,
            config,
        )
    except FilterError as exc:
        logger.exception("Filter stage failed: %s", exc)
        return 1

    print(f"cleaned_transcripts: {cleaned_path}")
    print(f"accepted: {accepted_path}")
    print(f"rejected: {rejected_path}")
    print(f"filter_report: {report_path}")
    print(f"total_filter_rows: {filter_stats['total_rows']}")
    print(f"accepted_count: {filter_stats['accepted_count']}")
    print(f"rejected_count: {filter_stats['rejected_count']}")
    return 0


def run_export_stage(accepted_path: Path, config: AppConfig) -> int:
    logger = logging.getLogger(config.project_name)
    output_dir = accepted_path.parent
    metadata_path = output_dir / "metadata.csv"
    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "summary.json"
    try:
        export_stats = export_dataset_from_accepted(
            accepted_path,
            metadata_path,
            manifest_path,
            summary_path,
            config,
        )
    except ExporterError as exc:
        logger.exception("Export stage failed: %s", exc)
        return 1

    print(f"accepted: {accepted_path}")
    print(f"metadata_csv: {metadata_path}")
    print(f"manifest_jsonl: {manifest_path}")
    print(f"summary_json: {summary_path}")
    print(f"total_accepted: {export_stats['total_accepted']}")
    print(f"total_duration_seconds: {export_stats['total_duration_seconds']}")
    print(f"total_duration_hours: {export_stats['total_duration_hours']}")
    return 0


def run_full_pipeline(args: argparse.Namespace, config: AppConfig) -> int:
    logger = logging.getLogger(config.project_name)
    working_dir = args.output if args.output is not None else config.paths.processed_dir

    try:
        outputs = preprocess_audio(
            input_path=args.input,
            working_dir=working_dir,
            config=config,
            logger=logger,
        )
        vad_segments = run_vad(outputs.audio_16k, config)
        vad_output_dir = ensure_dir(config.paths.output_dir / safe_stem(args.input))
        vad_output_path = save_vad_segments(
            vad_segments,
            vad_output_dir / "vad_segments.json",
        )
        loaded_vad_segments = load_vad_segments(vad_output_path)
        final_segments = build_segments(loaded_vad_segments, config)
        final_segments_path = save_final_segments(
            final_segments,
            vad_output_dir / "final_segments.json",
        )
        loaded_final_segments = load_final_segments(final_segments_path)
        manifest = cut_audio_segments(
            audio_path=outputs.audio_master,
            segments=loaded_final_segments,
            output_dir=vad_output_dir / "wavs",
        )
        manifest_path = save_json(manifest, vad_output_dir / "export_manifest.json")
        transcripts_output_path = vad_output_dir / "raw_transcripts.jsonl"
        logger.info(
            "Starting transcription with incremental JSONL output at %s",
            transcripts_output_path,
        )
        transcript_results = transcribe_all_segments(
            manifest_path,
            config,
            output_path=transcripts_output_path,
        )
        transcripts_path = transcripts_output_path
        logger.info(
            "Transcription and raw transcript save finished: %s",
            transcripts_path,
        )
        cleaned_transcripts_path = vad_output_dir / "cleaned_transcripts.jsonl"
        cleaning_stats = clean_all_transcripts(
            transcripts_path,
            cleaned_transcripts_path,
            config,
        )
        accepted_path = vad_output_dir / "accepted.jsonl"
        rejected_path = vad_output_dir / "rejected.jsonl"
        report_path = vad_output_dir / "filter_report.json"
        filter_stats = filter_all(
            cleaned_transcripts_path,
            accepted_path,
            rejected_path,
            report_path,
            config,
        )
        metadata_path = vad_output_dir / "metadata.csv"
        dataset_manifest_path = vad_output_dir / "manifest.jsonl"
        summary_path = vad_output_dir / "summary.json"
        export_stats = export_dataset_from_accepted(
            accepted_path,
            metadata_path,
            dataset_manifest_path,
            summary_path,
            config,
        )
    except (
        PreprocessError,
        VadError,
        SegmenterError,
        ExporterError,
        TranscriptionError,
        CleanerError,
        FilterError,
    ) as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1

    transcript_success_count = sum(
        1 for item in transcript_results if item.get("error") is None
    )
    transcript_failed_count = len(transcript_results) - transcript_success_count

    print(f"audio_16k: {outputs.audio_16k}")
    print(f"audio_master: {outputs.audio_master}")
    print(f"vad_segments: {vad_output_path}")
    print(f"final_segments: {final_segments_path}")
    print(f"export_manifest: {manifest_path}")
    print(f"raw_transcripts: {transcripts_path}")
    print(f"cleaned_transcripts: {cleaned_transcripts_path}")
    print(f"accepted: {accepted_path}")
    print(f"rejected: {rejected_path}")
    print(f"filter_report: {report_path}")
    print(f"metadata_csv: {metadata_path}")
    print(f"manifest_jsonl: {dataset_manifest_path}")
    print(f"summary_json: {summary_path}")
    print(f"raw_vad_segment_count: {len(vad_segments)}")
    print(f"final_segment_count: {len(final_segments)}")
    print(f"exported_segment_count: {len(manifest)}")
    print(f"successful_transcript_count: {transcript_success_count}")
    print(f"failed_transcript_count: {transcript_failed_count}")
    print(f"total_transcript_rows: {cleaning_stats['total_rows']}")
    print(f"cleaned_rows_count: {cleaning_stats['cleaned_rows']}")
    print(f"empty_rows_after_cleaning_count: {cleaning_stats['empty_rows_after_cleaning']}")
    print(f"accepted_count: {filter_stats['accepted_count']}")
    print(f"rejected_count: {filter_stats['rejected_count']}")
    print(f"total_accepted: {export_stats['total_accepted']}")
    print(f"total_duration_seconds: {export_stats['total_duration_seconds']}")
    print(f"total_duration_hours: {export_stats['total_duration_hours']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
