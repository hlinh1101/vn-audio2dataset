# vn-audio2dataset

`vn-audio2dataset` is a Python 3.11+ project scaffold for turning one long audio file from a single Vietnamese speaker into a clean TTS dataset.

Planned pipeline:

1. Preprocess audio
2. Run voice activity detection
3. Build natural 3-10 second speech segments
4. Export WAV segments
5. Run audio quality gating
6. Transcribe speech
7. Clean text
8. Filter low-quality samples
9. Export `metadata.csv`
10. Consolidate selected output folders into final fine-tuning datasets

## Current Status

This repository currently contains the base infrastructure, audio preprocessing, raw voice activity detection, final segment metadata building, segment WAV export, audio quality gating, raw transcription, transcript cleaning, quality filtering, final dataset metadata export, and explicit dataset consolidation:

- YAML configuration loading into structured dataclasses
- Reusable logging setup
- Small filesystem and JSON utilities
- CLI argument parsing
- Data directory layout for future pipeline stages
- FFmpeg-based conversion from `mp3`, `wav`, `m4a`, or `flac` into preprocessing WAV files
- Silero VAD speech-span detection saved as JSON
- Deterministic segment metadata builder for 3-10 second TTS candidates
- WAV export for final segments with duration validation and an export manifest
- Rule-based audio quality gating before transcription with `good`, `review`, and `bad` manifests
- Optional faster-whisper or ElevenLabs Scribe v2 transcription for exported WAV segments, saved as JSONL
- Optional ElevenLabs timestamp-first workflow for dominant-speaker TTS dataset building
- Rule-based Vietnamese-friendly transcript normalization saved as JSONL
- Rule-based quality filtering with accepted/rejected JSONL outputs and a report
- Final TTS dataset export with `metadata.csv`, `manifest.jsonl`, and `summary.json`
- Explicit consolidation of selected per-input output folders into one fine-tuning dataset

Advanced dataset packaging is intentionally not implemented yet.

## Project Structure

```text
vn-audio2dataset/
|-- README.md
|-- requirements.txt
|-- config.yaml
|-- main.py
|
+-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- logger.py
|   |-- utils.py
|   |-- preprocess.py
|   |-- vad.py
|   |-- segmenter.py
|   |-- audio_quality.py
|   |-- transcribe.py
|   |-- stt_workflow.py
|   |-- cleaner.py
|   |-- filter.py
|   |-- exporter.py
|   `-- consolidator.py
|
`-- data/
    |-- raw/
    |-- processed/
    |-- output/
    `-- rejects/
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

FFmpeg must also be installed and available on `PATH`.

The default transcription backend is local `faster_whisper`. To use ElevenLabs
Scribe v2 instead, set an API key in your environment or in a repo-root `.env`
file:

```text
ELEVENLABS_API_KEY=your_api_key_here
```

Then set the backend in `config.yaml`:

```yaml
transcription_backend: elevenlabs
```

Supported values are `faster_whisper` and `elevenlabs`. The ElevenLabs backend
uses `speech_to_text.convert` with `model_id="scribe_v2"` and
`language_code="vi"`. If `raw_transcripts.jsonl` already exists, completed
records are reused and only missing or previously failed segments are
transcribed.

## Usage

Run the full pipeline:

```bash
python main.py --input data/raw/sample.mp3
```

The command runs preprocessing, VAD, segment metadata building, WAV export, audio quality gating, transcription, transcript cleaning, quality filtering, and final dataset metadata export.

To run with ElevenLabs for a single command, edit `config.yaml` as shown above
or use a copied config file:

```bash
python main.py --config config.elevenlabs.yaml --input data/raw/sample.mp3
```

Preprocessing writes:

```text
data/processed/sample/audio_16k.wav
data/processed/sample/audio_master.wav
```

VAD writes:

```text
data/output/sample/vad_segments.json
```

Segment building writes:

```text
data/output/sample/final_segments.json
```

Audio export writes:

```text
data/output/sample/wavs/000001.wav
data/output/sample/export_manifest.json
```

Audio quality gating writes:

```text
data/output/sample/audio_quality_good.json
data/output/sample/audio_quality_review.json
data/output/sample/audio_quality_bad.json
data/output/sample/audio_quality_report.json
```

Transcription writes:

```text
data/output/sample/raw_transcripts.jsonl
```

Cleaning writes:

```text
data/output/sample/cleaned_transcripts.jsonl
```

Filtering writes:

```text
data/output/sample/accepted.jsonl
data/output/sample/rejected.jsonl
data/output/sample/filter_report.json
```

Final dataset export writes:

```text
data/output/sample/metadata.csv
data/output/sample/manifest.jsonl
data/output/sample/summary.json
```

Run the transcript-first ElevenLabs workflow:

```bash
python main.py --stage stt-full --input data/raw/sample.mp3
```

This mode sends the full preprocessed 16 kHz audio to ElevenLabs Scribe v2 with
word timestamps and diarization, selects the configured or automatically chosen
target speaker, builds timestamp-aligned clips, and then reuses the existing
audio quality, cleaning, filtering, and dataset export stages. It is intentionally conservative for
TTS/voice cloning: mixed-speaker spans, audio events, missing speaker IDs,
risky speaker transitions, and uncertain segment boundaries are rejected.
The timestamp segmenter scores semantic boundaries instead of cutting at a hard
target duration: it prefers sentence endings marked by `.`, `?`, `!`, or `…`
(`...` is also treated as repeated periods), does not split on commas, protects
Vietnamese-style named entities and connector phrases, and keeps an overlong
sentence as one exported segment.

By default, only `audio_quality_good.json` clips continue into
`raw_transcripts.jsonl`. To also include borderline `audio_quality_review.json`
clips while still excluding `audio_quality_bad.json`, add:

```bash
python main.py --stage stt-full --input data/raw/sample.mp3 --stt-include-review
```

If `data/output/sample_stt/elevenlabs_full_transcript.json` already exists,
`stt-full` reuses it by default and does not call the ElevenLabs API again. To
refresh the full-audio transcription cache, pass `--force`:

```bash
python main.py --stage stt-full --input data/raw/sample.mp3 --force
```

Inspect diarized speakers before exporting a dataset:

```bash
python main.py --stage stt-inspect --input data/raw/sample.mp3
python main.py --stage stt-inspect --input-dir data/raw/
```

Inspection mode uses the same full-transcript cache as `stt-full`, writes
speaker diagnostics, and stops before segmentation, audio quality filtering,
cleaning, and dataset export. Use the generated speaker IDs such as
`speaker_0` or `speaker_1` in `config.yaml` when you want a specific anchor,
MC, reporter, or interview speaker.

Run `stt-full` over a folder:

```bash
python main.py --stage stt-full --input-dir data/raw/
```

Folder mode recursively finds supported audio files using the same formats as
preprocessing: `mp3`, `wav`, `m4a`, and `flac`. Each file is processed
independently and writes to its own normal `data/output/<stem>_stt/` folder.
If one file fails, the batch logs the error and continues with the remaining
files.

By default, folder mode skips files whose output folder already contains a
complete final dataset export (`summary.json`, `metadata.csv`, `manifest.jsonl`,
and `accepted.jsonl`). Use `--force` to reprocess complete outputs and refresh
the cached ElevenLabs full transcript:

```bash
python main.py --stage stt-full --input-dir data/raw/ --force
```

The final batch summary prints:

```text
batch_total_files
batch_succeeded
batch_failed
batch_skipped
batch_total_exported_duration_seconds
batch_total_exported_duration_hours
```

`stt-full` writes to a separate output folder so the VAD-first pipeline remains
untouched:

```text
data/output/sample_stt/elevenlabs_full_transcript.json
data/output/sample_stt/stt_words.json
data/output/sample_stt/stt_speaker_inspection.json
data/output/sample_stt/stt_speaker_turns.json
data/output/sample_stt/stt_segments.json
data/output/sample_stt/stt_rejected_segments.json
data/output/sample_stt/stt_boundary_scores.json
data/output/sample_stt/stt_segmentation_report.json
data/output/sample_stt/wavs/000001.wav
data/output/sample_stt/export_manifest.json
data/output/sample_stt/audio_quality_good.json
data/output/sample_stt/raw_transcripts.jsonl
data/output/sample_stt/cleaned_transcripts.jsonl
data/output/sample_stt/accepted.jsonl
data/output/sample_stt/metadata.csv
data/output/sample_stt/manifest.jsonl
data/output/sample_stt/summary.json
```

Configure its conservative timestamp segmentation behavior in `config.yaml`:

```yaml
stt_segmentation:
  model_id: scribe_v2
  timestamps_granularity: word
  diarize: true
  tag_audio_events: true
  dominant_speaker: auto
  speaker_selection_mode: auto
  default_target_speaker: auto
  require_manual_speaker: false
  per_file_target_speakers: {}
  min_speaker_share: 0.45
  boundary_pad_sec: 0.05
  boundary_guard_sec: 0.2
  max_word_gap_sec: 0.8
  preferred_min_sec: 5.0
  preferred_max_sec: 10.0
  semantic_max_sec: 15.0
  sentence_punctuation_weight: 8.0
  # Backward-compatible legacy key; commas/clauses are not STT boundaries.
  clause_punctuation_weight: 4.0
  pause_strong_sec: 0.45
  pause_medium_sec: 0.25
  pause_strong_weight: 5.0
  pause_medium_weight: 2.5
  min_boundary_score: 7.0
  allow_short_clips: false
  protect_named_entities: true
  protect_connector_phrases: true
  min_words: 3
  min_avg_logprob:
```

For multi-speaker news or broadcast audio, configure target speakers per file
after running `stt-inspect`:

```yaml
stt_segmentation:
  speaker_selection_mode: manual
  require_manual_speaker: true
  per_file_target_speakers:
    sample: speaker_1
    data/raw/news_show.wav: speaker_2
```

Mapping lookup accepts the exact path, normalized relative path, file name, or
file stem. In `auto` mode, low `min_speaker_share` no longer fails the run by
itself; it is reported as a speaker-selection warning.

Run only transcript cleaning from an existing raw transcript file:

```bash
python main.py --stage clean --raw data/output/sample/raw_transcripts.jsonl
```

Clean-only mode skips preprocessing, VAD, segmentation, WAV export, and transcription. It writes:

```text
data/output/sample/cleaned_transcripts.jsonl
```

Run only quality filtering from an existing cleaned transcript file:

```bash
python main.py --stage filter --cleaned data/output/sample/cleaned_transcripts.jsonl
```

Filter-only mode skips preprocessing, VAD, segmentation, WAV export, transcription, and cleaning. It writes:

```text
data/output/sample/accepted.jsonl
data/output/sample/rejected.jsonl
data/output/sample/filter_report.json
```

The filter keeps `word_count` and `char_count` on each output row for review.
Character count is diagnostic only; it is not a hard rejection reason. Duration
remains the primary length gate, controlled by `filter.min_sec` and
`filter.max_sec`.

Run only audio quality gating from an existing exported manifest:

```bash
python main.py --stage quality --manifest data/output/sample/export_manifest.json
```

Quality-only mode skips preprocessing, VAD, segmentation, WAV export, transcription, cleaning, filtering, and final dataset export. It writes:

```text
data/output/sample/audio_quality_good.json
data/output/sample/audio_quality_review.json
data/output/sample/audio_quality_bad.json
data/output/sample/audio_quality_report.json
```

Run only final dataset metadata export from an existing accepted file:

```bash
python main.py --stage export --accepted data/output/sample/accepted.jsonl
```

Export-only mode skips preprocessing, VAD, segmentation, WAV export, transcription, cleaning, and filtering. It writes:

```text
data/output/sample/metadata.csv
data/output/sample/manifest.jsonl
data/output/sample/summary.json
```

Consolidate selected output folders into one final fine-tuning dataset:

```bash
python main.py --stage consolidate --output data/output/datasets/speaker_a ^
  --source data/output/sample_001 ^
  --source data/output/sample_002
```

Consolidate mode never auto-discovers folders. Only folders passed with `--source`
are merged, so you can build separate datasets for different speakers or
experiments. Each source folder must contain `accepted.jsonl` and the referenced
WAV files. It copies the selected audio into the final dataset folder and writes:

```text
data/output/datasets/speaker_a/wavs/
data/output/datasets/speaker_a/accepted.jsonl
data/output/datasets/speaker_a/metadata.csv
data/output/datasets/speaker_a/manifest.jsonl
data/output/datasets/speaker_a/summary.json
data/output/datasets/speaker_a/consolidation_report.json
```

Copied WAV names are prefixed with the source folder name to avoid collisions
between folders that each contain files such as `000001.wav`.

Use `--output` to override the processed working directory in full mode. In
consolidate mode, `--output` is the final dataset directory.

## Audio Quality Rules

The audio quality step is intentionally rule-based. It scores each exported WAV with simple, explainable proxies:

- duration range
- RMS loudness
- clipping ratio
- internal silence ratio
- leading and trailing silence
- spectral flatness and high-frequency energy as conservative noise/reverb proxies

It writes three manifests:

- `audio_quality_good.json`: safe to pass to transcription
- `audio_quality_review.json`: borderline segments worth manual review
- `audio_quality_bad.json`: clearly poor or invalid segments

The original audio files and the original `export_manifest.json` are left unchanged.

## Transcript-First Limitations

- Input must already be a local audio file; this workflow does not download
  podcasts, videos, or URLs.
- Speaker isolation is diarization-based selection and rejection, not source
  separation.
- Very conversational material may produce few accepted clips because speaker
  overlap and close turn-taking are discarded aggressively.
- ElevenLabs API limits, pricing, and maximum upload duration still apply.
