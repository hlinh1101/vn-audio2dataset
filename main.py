"""CLI entrypoint for vn-audio2dataset."""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

from src.audio_quality import AudioQualityError, filter_audio_manifest
from src.cleaner import CleanerError, clean_all_transcripts
from src.config import AppConfig, load_config
from src.consolidator import ConsolidationError, consolidate_output_folders
from src.exporter import (
    ExporterError,
    cut_audio_segments,
    export_dataset_from_accepted,
    load_final_segments,
)
from src.filter import FilterError, filter_all
from src.logger import setup_logger
from src.preprocess import SUPPORTED_AUDIO_EXTENSIONS, PreprocessError, preprocess_audio
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
from src.stt_workflow import (
    SttWorkflowError,
    build_speaker_inspection,
    build_timestamp_segments,
    save_full_stt_response,
    transcribe_full_audio_with_elevenlabs,
    write_transcripts_from_stt_segments,
)
from src.utils import ensure_dir, load_json, safe_stem, save_json
from src.vad import VadError, run_vad, save_vad_segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess audio, run Silero VAD, build segment metadata, "
            "export segment WAV files, transcribe them, clean text, filter quality, "
            "export final dataset metadata, and consolidate selected outputs."
        )
    )
    parser.add_argument(
        "--stage",
        choices=(
            "full",
            "stt-full",
            "stt-inspect",
            "clean",
            "quality",
            "filter",
            "export",
            "consolidate",
        ),
        default="full",
        help=(
            "Pipeline stage to run. Use 'stt-full' for ElevenLabs timestamp-first "
            "segmentation, 'stt-inspect' for diarization inspection only, 'clean' "
            "for existing raw transcripts, 'quality' for exported audio manifests, "
            "'filter' for existing cleaned transcripts, 'export' for accepted "
            "records, or 'consolidate' for explicitly selected output folders."
        ),
    )
    parser.add_argument(
        "--input",
        required=False,
        type=Path,
        help="Path to the long input audio file.",
    )
    parser.add_argument(
        "--input-dir",
        required=False,
        type=Path,
        help="Directory of audio files to process with --stage stt-full or stt-inspect.",
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
        "--manifest",
        required=False,
        type=Path,
        help="Path to export_manifest.json for audio-quality-only mode.",
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
        help=(
            "Processed working directory for full or stt-full mode, or final "
            "dataset directory for consolidate mode. Defaults to paths.processed_dir "
            "from config."
        ),
    )
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        dest="sources",
        help=(
            "Output folder to include in consolidate mode. Repeat this option for "
            "each folder to merge."
        ),
    )
    parser.add_argument(
        "--stt-include-review",
        action="store_true",
        help=(
            "For --stage stt-full only, include audio_quality_review clips in "
            "downstream raw_transcripts.jsonl. audio_quality_bad clips are still "
            "excluded."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "For --stage stt-full or stt-inspect only, reprocess complete outputs "
            "and ignore an existing elevenlabs_full_transcript.json."
        ),
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
    if args.stage == "quality" and args.manifest is None:
        parser.error("--manifest is required when --stage quality is used.")
    if args.stage == "filter" and args.cleaned is None:
        parser.error("--cleaned is required when --stage filter is used.")
    if args.stage == "export" and args.accepted is None:
        parser.error("--accepted is required when --stage export is used.")
    if args.stage == "consolidate":
        if not args.sources:
            parser.error("--source is required when --stage consolidate is used.")
        if args.output is None:
            parser.error("--output is required when --stage consolidate is used.")
    if args.stage in {"stt-full", "stt-inspect"}:
        if (args.input is None) == (args.input_dir is None):
            parser.error(
                "Exactly one of --input or --input-dir is required when "
                "--stage stt-full or --stage stt-inspect is used."
            )
    elif args.input_dir is not None:
        parser.error("--input-dir can only be used with --stage stt-full or stt-inspect.")
    if args.stage == "full" and args.input is None:
        parser.error(
            "--input is required unless --stage stt-full, --stage clean, "
            "--stage quality, --stage filter, --stage export, or "
            "--stage consolidate is used."
        )
    if args.stt_include_review and args.stage != "stt-full":
        parser.error("--stt-include-review can only be used with --stage stt-full.")
    if args.force and args.stage not in {"stt-full", "stt-inspect"}:
        parser.error("--force can only be used with --stage stt-full or stt-inspect.")
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
    if args.stage == "quality":
        return run_quality_stage(args.manifest, config)
    if args.stage == "filter":
        return run_filter_stage(args.cleaned, config)
    if args.stage == "export":
        return run_export_stage(args.accepted, config)
    if args.stage == "consolidate":
        return run_consolidate_stage(args.sources, args.output, config)
    if args.stage == "stt-full":
        return run_stt_full_pipeline(args, config)
    if args.stage == "stt-inspect":
        return run_stt_inspect_pipeline(args, config)

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


def run_quality_stage(manifest_path: Path, config: AppConfig) -> int:
    logger = logging.getLogger(config.project_name)
    output_dir = manifest_path.parent
    good_path = output_dir / "audio_quality_good.json"
    bad_path = output_dir / "audio_quality_bad.json"
    review_path = output_dir / "audio_quality_review.json"
    report_path = output_dir / "audio_quality_report.json"
    try:
        quality_stats = filter_audio_manifest(
            manifest_path,
            good_path,
            bad_path,
            review_path,
            report_path,
            config,
        )
    except AudioQualityError as exc:
        logger.exception("Audio quality stage failed: %s", exc)
        return 1

    print(f"export_manifest: {manifest_path}")
    print(f"audio_quality_good: {good_path}")
    print(f"audio_quality_review: {review_path}")
    print(f"audio_quality_bad: {bad_path}")
    print(f"audio_quality_report: {report_path}")
    print(f"audio_quality_good_count: {quality_stats['good_count']}")
    print(f"audio_quality_review_count: {quality_stats['review_count']}")
    print(f"audio_quality_bad_count: {quality_stats['bad_count']}")
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


def run_consolidate_stage(
    source_dirs: list[Path],
    output_dir: Path,
    config: AppConfig,
) -> int:
    logger = logging.getLogger(config.project_name)
    try:
        consolidation_stats = consolidate_output_folders(
            source_dirs=source_dirs,
            output_dir=output_dir,
            config=config,
        )
    except ConsolidationError as exc:
        logger.exception("Consolidate stage failed: %s", exc)
        return 1

    print(f"output_dir: {consolidation_stats['output_dir']}")
    print(f"wavs_dir: {consolidation_stats['wavs_dir']}")
    print(f"accepted: {consolidation_stats['accepted_path']}")
    print(f"metadata_csv: {consolidation_stats['metadata_path']}")
    print(f"manifest_jsonl: {consolidation_stats['manifest_path']}")
    print(f"summary_json: {consolidation_stats['summary_path']}")
    print(f"consolidation_report: {consolidation_stats['report_path']}")
    print(f"source_count: {consolidation_stats['source_count']}")
    print(f"total_accepted: {consolidation_stats['total_accepted']}")
    print(f"total_duration_seconds: {consolidation_stats['total_duration_seconds']}")
    print(f"total_duration_hours: {consolidation_stats['total_duration_hours']}")
    return 0


def run_stt_full_pipeline(args: argparse.Namespace, config: AppConfig) -> int:
    if args.input_dir is not None:
        return run_stt_full_batch(args, config)

    result = run_stt_full_file(
        input_path=args.input,
        args=args,
        config=config,
    )
    return 0 if result["status"] == "succeeded" else 1


def run_stt_inspect_pipeline(args: argparse.Namespace, config: AppConfig) -> int:
    if args.input_dir is not None:
        return run_stt_inspect_batch(args, config)

    result = run_stt_inspect_file(args.input, args, config)
    return 0 if result["status"] == "succeeded" else 1


def run_stt_inspect_file(
    input_path: Path,
    args: argparse.Namespace,
    config: AppConfig,
) -> dict[str, object]:
    logger = logging.getLogger(config.project_name)
    working_dir = args.output if args.output is not None else config.paths.processed_dir
    stt_output_dir = ensure_dir(_stt_output_dir_for_input(input_path, config))

    try:
        outputs = preprocess_audio(
            input_path=input_path,
            working_dir=working_dir,
            config=config,
            logger=logger,
        )
        response, full_response_path, used_cached_full_response = _load_or_create_full_stt_response(
            audio_16k_path=outputs.audio_16k,
            stt_output_dir=stt_output_dir,
            args=args,
            config=config,
        )
        inspection_paths = _save_stt_speaker_inspection(response, stt_output_dir, config)
        inspection = inspection_paths["speaker_inspection"]
    except (PreprocessError, SttWorkflowError, TranscriptionError) as exc:
        logger.exception("STT inspection failed for %s: %s", input_path, exc)
        return {
            "status": "failed",
            "input_path": str(input_path),
            "output_dir": str(stt_output_dir),
            "error": str(exc),
        }

    print(f"audio_16k: {outputs.audio_16k}")
    print(f"stt_output_dir: {stt_output_dir}")
    print(f"elevenlabs_full_transcript: {full_response_path}")
    print(f"used_cached_full_transcript: {used_cached_full_response}")
    print(f"force_full_transcript_refresh: {args.force}")
    print(f"stt_words: {inspection_paths['stt_words_path']}")
    print(f"stt_speaker_inspection: {inspection_paths['speaker_inspection_path']}")
    print(f"stt_speaker_turns: {inspection_paths['speaker_turns_path']}")
    print(f"speaker_count: {inspection['speaker_count']}")
    for speaker_id, speaker in inspection["speakers"].items():
        print(
            "speaker_summary: "
            f"{speaker_id} duration={speaker['duration_seconds']}s "
            f"share={speaker['share']} words={speaker['word_count']} "
            f"turns={speaker['turn_count']}"
        )
    return {
        "status": "succeeded",
        "input_path": str(input_path),
        "output_dir": str(stt_output_dir),
        "speaker_count": int(inspection["speaker_count"]),
    }


def run_stt_inspect_batch(args: argparse.Namespace, config: AppConfig) -> int:
    logger = logging.getLogger(config.project_name)
    input_dir = args.input_dir
    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        print(f"inspect_input_dir: {input_dir}")
        print("inspect_error: input_directory_missing")
        return 1
    if not input_dir.is_dir():
        logger.error("Input directory path is not a directory: %s", input_dir)
        print(f"inspect_input_dir: {input_dir}")
        print("inspect_error: input_path_not_directory")
        return 1

    audio_files = _find_supported_audio_files(input_dir)
    results: list[dict[str, object]] = []
    seen_output_dirs: dict[Path, Path] = {}
    logger.info("Found %d supported audio files under %s", len(audio_files), input_dir)

    for index, audio_path in enumerate(audio_files, start=1):
        output_dir = _stt_output_dir_for_input(audio_path, config)
        print(f"inspect_progress: {index}/{len(audio_files)}")
        print(f"inspect_input_file: {audio_path}")
        print(f"inspect_output_dir: {output_dir}")

        previous_input = seen_output_dirs.get(output_dir)
        if previous_input is not None:
            message = f"output_dir_collision with {previous_input}"
            logger.error("Skipping %s: %s", audio_path, message)
            print("inspect_file_status: failed")
            print(f"inspect_file_error: {message}")
            results.append({"status": "failed", "error": message})
            continue
        seen_output_dirs[output_dir] = audio_path

        try:
            result = run_stt_inspect_file(audio_path, args, config)
        except Exception as exc:
            logger.exception("Unexpected STT inspection failure for %s: %s", audio_path, exc)
            result = {"status": "failed", "error": str(exc)}
        print(f"inspect_file_status: {result['status']}")
        if result["status"] == "failed":
            print(f"inspect_file_error: {result.get('error', '')}")
        results.append(result)

    succeeded = sum(1 for item in results if item["status"] == "succeeded")
    failed = sum(1 for item in results if item["status"] == "failed")
    print(f"inspect_input_dir: {input_dir}")
    print(f"inspect_total_files: {len(audio_files)}")
    print(f"inspect_succeeded: {succeeded}")
    print(f"inspect_failed: {failed}")
    return 1 if failed else 0


def run_stt_full_file(
    input_path: Path,
    args: argparse.Namespace,
    config: AppConfig,
) -> dict[str, object]:
    logger = logging.getLogger(config.project_name)
    working_dir = args.output if args.output is not None else config.paths.processed_dir
    stt_output_dir = ensure_dir(_stt_output_dir_for_input(input_path, config))

    try:
        outputs = preprocess_audio(
            input_path=input_path,
            working_dir=working_dir,
            config=config,
            logger=logger,
        )
        response, full_response_path, used_cached_full_response = _load_or_create_full_stt_response(
            audio_16k_path=outputs.audio_16k,
            stt_output_dir=stt_output_dir,
            args=args,
            config=config,
        )
        inspection_paths = _save_stt_speaker_inspection(response, stt_output_dir, config)
        segmentation = build_timestamp_segments(response, config, input_path=input_path)
        stt_words_path = save_json(segmentation["words"], stt_output_dir / "stt_words.json")
        stt_segments_path = save_json(
            segmentation["segments"],
            stt_output_dir / "stt_segments.json",
        )
        stt_rejected_path = save_json(
            segmentation["rejected_segments"],
            stt_output_dir / "stt_rejected_segments.json",
        )
        stt_boundary_scores_path = save_json(
            segmentation["boundary_scores"],
            stt_output_dir / "stt_boundary_scores.json",
        )
        stt_report_path = save_json(
            segmentation["report"],
            stt_output_dir / "stt_segmentation_report.json",
        )
        if not segmentation["segments"]:
            raise SttWorkflowError(
                "STT timestamp segmentation accepted 0 segments. "
                f"Review {stt_report_path} before relaxing thresholds."
            )

        manifest = cut_audio_segments(
            audio_path=outputs.audio_master,
            segments=segmentation["segments"],
            output_dir=stt_output_dir / "wavs",
        )
        manifest_path = save_json(manifest, stt_output_dir / "export_manifest.json")
        audio_quality_good_path = stt_output_dir / "audio_quality_good.json"
        audio_quality_bad_path = stt_output_dir / "audio_quality_bad.json"
        audio_quality_review_path = stt_output_dir / "audio_quality_review.json"
        audio_quality_report_path = stt_output_dir / "audio_quality_report.json"
        quality_config = copy.deepcopy(config)
        quality_config.audio_quality.max_sec = max(
            quality_config.audio_quality.max_sec,
            config.stt_segmentation.semantic_max_sec,
        )
        quality_stats = filter_audio_manifest(
            manifest_path,
            audio_quality_good_path,
            audio_quality_bad_path,
            audio_quality_review_path,
            audio_quality_report_path,
            quality_config,
        )
        if quality_stats["good_count"] == 0 and not args.stt_include_review:
            raise AudioQualityError(
                "Audio quality stage accepted 0 STT-derived segments. "
                f"Review {audio_quality_report_path} before relaxing thresholds."
            )

        good_manifest = load_json(audio_quality_good_path)
        if not isinstance(good_manifest, list):
            raise SttWorkflowError(
                f"Audio quality good manifest must contain a list: {audio_quality_good_path}"
            )
        transcript_manifest = list(good_manifest)
        review_manifest: list[dict[str, object]] = []
        if args.stt_include_review:
            raw_review_manifest = load_json(audio_quality_review_path)
            if not isinstance(raw_review_manifest, list):
                raise SttWorkflowError(
                    "Audio quality review manifest must contain a list: "
                    f"{audio_quality_review_path}"
                )
            review_manifest = raw_review_manifest
            transcript_manifest.extend(review_manifest)
            transcript_manifest.sort(
                key=lambda item: (
                    float(item.get("start", 0.0)) if isinstance(item, dict) else 0.0,
                    str(item.get("id", "")) if isinstance(item, dict) else "",
                )
            )
        if not transcript_manifest:
            raise AudioQualityError(
                "Audio quality stage produced 0 usable STT-derived segments for "
                "raw transcript export. Review the audio quality outputs before "
                "relaxing thresholds."
            )

        transcripts_path = stt_output_dir / "raw_transcripts.jsonl"
        transcript_results = write_transcripts_from_stt_segments(
            transcript_manifest,
            segmentation["segments"],
            transcripts_path,
            config,
        )
        cleaned_transcripts_path = stt_output_dir / "cleaned_transcripts.jsonl"
        cleaning_stats = clean_all_transcripts(
            transcripts_path,
            cleaned_transcripts_path,
            config,
        )
        accepted_path = stt_output_dir / "accepted.jsonl"
        rejected_path = stt_output_dir / "rejected.jsonl"
        report_path = stt_output_dir / "filter_report.json"
        filter_stats = filter_all(
            cleaned_transcripts_path,
            accepted_path,
            rejected_path,
            report_path,
            config,
        )
        metadata_path = stt_output_dir / "metadata.csv"
        dataset_manifest_path = stt_output_dir / "manifest.jsonl"
        summary_path = stt_output_dir / "summary.json"
        export_stats = export_dataset_from_accepted(
            accepted_path,
            metadata_path,
            dataset_manifest_path,
            summary_path,
            config,
        )
    except (
        PreprocessError,
        SttWorkflowError,
        AudioQualityError,
        ExporterError,
        CleanerError,
        FilterError,
        TranscriptionError,
    ) as exc:
        logger.exception("STT-first pipeline failed for %s: %s", input_path, exc)
        return {
            "status": "failed",
            "input_path": str(input_path),
            "output_dir": str(stt_output_dir),
            "error": str(exc),
            "total_duration_seconds": 0.0,
        }

    print(f"audio_16k: {outputs.audio_16k}")
    print(f"audio_master: {outputs.audio_master}")
    print(f"stt_output_dir: {stt_output_dir}")
    print(f"elevenlabs_full_transcript: {full_response_path}")
    print(f"used_cached_full_transcript: {used_cached_full_response}")
    print(f"force_full_transcript_refresh: {args.force}")
    print(f"stt_words: {stt_words_path}")
    print(f"stt_speaker_inspection: {inspection_paths['speaker_inspection_path']}")
    print(f"stt_speaker_turns: {inspection_paths['speaker_turns_path']}")
    print(f"stt_segments: {stt_segments_path}")
    print(f"stt_rejected_segments: {stt_rejected_path}")
    print(f"stt_boundary_scores: {stt_boundary_scores_path}")
    print(f"stt_segmentation_report: {stt_report_path}")
    print(f"export_manifest: {manifest_path}")
    print(f"audio_quality_good: {audio_quality_good_path}")
    print(f"audio_quality_review: {audio_quality_review_path}")
    print(f"audio_quality_bad: {audio_quality_bad_path}")
    print(f"audio_quality_report: {audio_quality_report_path}")
    print(f"raw_transcripts: {transcripts_path}")
    print(f"cleaned_transcripts: {cleaned_transcripts_path}")
    print(f"accepted: {accepted_path}")
    print(f"rejected: {rejected_path}")
    print(f"filter_report: {report_path}")
    print(f"metadata_csv: {metadata_path}")
    print(f"manifest_jsonl: {dataset_manifest_path}")
    print(f"summary_json: {summary_path}")
    print(f"stt_word_count: {len(segmentation['words'])}")
    print(f"stt_segment_count: {len(segmentation['segments'])}")
    print(f"stt_rejected_segment_count: {len(segmentation['rejected_segments'])}")
    print(f"selected_speaker: {segmentation['report']['selected_speaker']}")
    print(f"exported_segment_count: {len(manifest)}")
    print(f"audio_quality_good_count: {quality_stats['good_count']}")
    print(f"audio_quality_review_count: {quality_stats['review_count']}")
    print(f"audio_quality_bad_count: {quality_stats['bad_count']}")
    print(f"stt_include_review: {args.stt_include_review}")
    print(f"raw_transcript_source_count: {len(transcript_manifest)}")
    print(f"raw_transcript_review_source_count: {len(review_manifest)}")
    print(f"raw_transcript_count: {len(transcript_results)}")
    print(f"total_transcript_rows: {cleaning_stats['total_rows']}")
    print(f"cleaned_rows_count: {cleaning_stats['cleaned_rows']}")
    print(f"empty_rows_after_cleaning_count: {cleaning_stats['empty_rows_after_cleaning']}")
    print(f"accepted_count: {filter_stats['accepted_count']}")
    print(f"rejected_count: {filter_stats['rejected_count']}")
    print(f"total_accepted: {export_stats['total_accepted']}")
    print(f"total_duration_seconds: {export_stats['total_duration_seconds']}")
    print(f"total_duration_hours: {export_stats['total_duration_hours']}")
    return {
        "status": "succeeded",
        "input_path": str(input_path),
        "output_dir": str(stt_output_dir),
        "summary_path": str(summary_path),
        "total_accepted": int(export_stats["total_accepted"]),
        "total_duration_seconds": float(export_stats["total_duration_seconds"]),
    }


def run_stt_full_batch(args: argparse.Namespace, config: AppConfig) -> int:
    logger = logging.getLogger(config.project_name)
    input_dir = args.input_dir
    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        print(f"batch_input_dir: {input_dir}")
        print("batch_error: input_directory_missing")
        return 1
    if not input_dir.is_dir():
        logger.error("Input directory path is not a directory: %s", input_dir)
        print(f"batch_input_dir: {input_dir}")
        print("batch_error: input_path_not_directory")
        return 1

    audio_files = _find_supported_audio_files(input_dir)
    logger.info(
        "Found %d supported audio files under %s",
        len(audio_files),
        input_dir,
    )

    results: list[dict[str, object]] = []
    seen_output_dirs: dict[Path, Path] = {}

    for index, audio_path in enumerate(audio_files, start=1):
        output_dir = _stt_output_dir_for_input(audio_path, config)
        logger.info(
            "STT batch file %d/%d: %s -> %s",
            index,
            len(audio_files),
            audio_path,
            output_dir,
        )
        print(f"batch_progress: {index}/{len(audio_files)}")
        print(f"batch_input_file: {audio_path}")
        print(f"batch_output_dir: {output_dir}")

        previous_input = seen_output_dirs.get(output_dir)
        if previous_input is not None:
            message = (
                "output_dir_collision with "
                f"{previous_input}; current output naming would not keep files separated"
            )
            logger.error("Skipping %s: %s", audio_path, message)
            print("batch_file_status: failed")
            print(f"batch_file_error: {message}")
            results.append(
                {
                    "status": "failed",
                    "input_path": str(audio_path),
                    "output_dir": str(output_dir),
                    "error": message,
                    "total_duration_seconds": 0.0,
                }
            )
            continue
        seen_output_dirs[output_dir] = audio_path

        completed_summary = _load_complete_stt_output_summary(output_dir)
        if completed_summary is not None and not args.force:
            duration = _summary_duration(completed_summary)
            logger.info("Skipping complete STT output for %s: %s", audio_path, output_dir)
            print("batch_file_status: skipped")
            print(f"batch_file_duration_seconds: {duration}")
            results.append(
                {
                    "status": "skipped",
                    "input_path": str(audio_path),
                    "output_dir": str(output_dir),
                    "summary_path": str(output_dir / "summary.json"),
                    "total_duration_seconds": duration,
                }
            )
            continue

        try:
            result = run_stt_full_file(audio_path, args, config)
        except Exception as exc:
            logger.exception("Unexpected STT batch failure for %s: %s", audio_path, exc)
            result = {
                "status": "failed",
                "input_path": str(audio_path),
                "output_dir": str(output_dir),
                "error": str(exc),
                "total_duration_seconds": 0.0,
            }
        print(f"batch_file_status: {result['status']}")
        if result["status"] == "failed":
            print(f"batch_file_error: {result.get('error', '')}")
        results.append(result)

    total_files = len(audio_files)
    succeeded = sum(1 for item in results if item["status"] == "succeeded")
    failed = sum(1 for item in results if item["status"] == "failed")
    skipped = sum(1 for item in results if item["status"] == "skipped")
    total_duration_seconds = round(
        sum(float(item.get("total_duration_seconds", 0.0) or 0.0) for item in results),
        3,
    )

    print(f"batch_input_dir: {input_dir}")
    print(f"batch_total_files: {total_files}")
    print(f"batch_succeeded: {succeeded}")
    print(f"batch_failed: {failed}")
    print(f"batch_skipped: {skipped}")
    print(f"batch_total_exported_duration_seconds: {total_duration_seconds}")
    print(f"batch_total_exported_duration_hours: {round(total_duration_seconds / 3600.0, 4)}")
    logger.info(
        "STT batch complete: total=%d succeeded=%d failed=%d skipped=%d duration=%.3fs",
        total_files,
        succeeded,
        failed,
        skipped,
        total_duration_seconds,
    )
    return 1 if failed else 0


def _find_supported_audio_files(input_dir: Path) -> list[Path]:
    supported = {extension.lower() for extension in SUPPORTED_AUDIO_EXTENSIONS}
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in supported
    )


def _load_or_create_full_stt_response(
    audio_16k_path: Path,
    stt_output_dir: Path,
    args: argparse.Namespace,
    config: AppConfig,
) -> tuple[object, Path, bool]:
    logger = logging.getLogger(config.project_name)
    full_response_path = stt_output_dir / "elevenlabs_full_transcript.json"
    if full_response_path.exists() and not args.force:
        try:
            response = load_json(full_response_path)
        except Exception as exc:
            raise SttWorkflowError(
                "Failed to read existing ElevenLabs full transcript cache "
                f"{full_response_path}. Re-run with --force to call the API again "
                f"and replace it: {exc}"
            ) from exc
        logger.info("Reusing existing ElevenLabs full transcript: %s", full_response_path)
        return response, full_response_path, True

    response = transcribe_full_audio_with_elevenlabs(audio_16k_path, config)
    return response, save_full_stt_response(response, full_response_path), False


def _save_stt_speaker_inspection(
    response: object,
    stt_output_dir: Path,
    config: AppConfig,
) -> dict[str, object]:
    inspection_result = build_speaker_inspection(response, config)
    stt_words_path = save_json(inspection_result["words"], stt_output_dir / "stt_words.json")
    speaker_inspection_path = save_json(
        inspection_result["speaker_inspection"],
        stt_output_dir / "stt_speaker_inspection.json",
    )
    speaker_turns_path = save_json(
        inspection_result["speaker_turns"],
        stt_output_dir / "stt_speaker_turns.json",
    )
    return {
        "stt_words_path": stt_words_path,
        "speaker_inspection_path": speaker_inspection_path,
        "speaker_turns_path": speaker_turns_path,
        "speaker_inspection": inspection_result["speaker_inspection"],
        "speaker_turns": inspection_result["speaker_turns"],
    }


def _stt_output_dir_for_input(input_path: Path, config: AppConfig) -> Path:
    return config.paths.output_dir / f"{safe_stem(input_path)}_stt"


def _load_complete_stt_output_summary(output_dir: Path) -> dict[str, object] | None:
    required_paths = [
        output_dir / "summary.json",
        output_dir / "metadata.csv",
        output_dir / "manifest.jsonl",
        output_dir / "accepted.jsonl",
    ]
    if not output_dir.exists() or not output_dir.is_dir():
        return None
    if any(not path.exists() or not path.is_file() for path in required_paths):
        return None

    try:
        summary = load_json(output_dir / "summary.json")
    except Exception:
        return None
    if not isinstance(summary, dict):
        return None
    if "total_duration_seconds" not in summary or "total_accepted" not in summary:
        return None
    try:
        float(summary["total_duration_seconds"])
        int(summary["total_accepted"])
    except (TypeError, ValueError):
        return None
    return summary


def _summary_duration(summary: dict[str, object]) -> float:
    try:
        return round(float(summary.get("total_duration_seconds", 0.0)), 3)
    except (TypeError, ValueError):
        return 0.0


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
        audio_quality_good_path = vad_output_dir / "audio_quality_good.json"
        audio_quality_bad_path = vad_output_dir / "audio_quality_bad.json"
        audio_quality_review_path = vad_output_dir / "audio_quality_review.json"
        audio_quality_report_path = vad_output_dir / "audio_quality_report.json"
        quality_stats = filter_audio_manifest(
            manifest_path,
            audio_quality_good_path,
            audio_quality_bad_path,
            audio_quality_review_path,
            audio_quality_report_path,
            config,
        )
        if quality_stats["good_count"] == 0:
            raise AudioQualityError(
                "Audio quality stage accepted 0 segments. "
                f"Review {audio_quality_report_path} before transcription."
            )
        transcripts_output_path = vad_output_dir / "raw_transcripts.jsonl"
        logger.info(
            "Starting transcription with incremental JSONL output at %s",
            transcripts_output_path,
        )
        transcript_results = transcribe_all_segments(
            audio_quality_good_path,
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
        AudioQualityError,
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
    print(f"audio_quality_good: {audio_quality_good_path}")
    print(f"audio_quality_review: {audio_quality_review_path}")
    print(f"audio_quality_bad: {audio_quality_bad_path}")
    print(f"audio_quality_report: {audio_quality_report_path}")
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
    print(f"audio_quality_good_count: {quality_stats['good_count']}")
    print(f"audio_quality_review_count: {quality_stats['review_count']}")
    print(f"audio_quality_bad_count: {quality_stats['bad_count']}")
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
