# vn-audio2dataset

`vn-audio2dataset` is a Python 3.11+ project scaffold for turning one long audio file from a single Vietnamese speaker into a clean TTS dataset.

Planned pipeline:

1. Preprocess audio
2. Run voice activity detection
3. Build natural 3-10 second speech segments
4. Export WAV segments
5. Transcribe speech
6. Clean text
7. Filter low-quality samples
8. Export `metadata.csv`

## Current Status

This repository currently contains the base infrastructure, audio preprocessing, raw voice activity detection, final segment metadata building, segment WAV export, raw transcription, transcript cleaning, quality filtering, and final dataset metadata export:

- YAML configuration loading into structured dataclasses
- Reusable logging setup
- Small filesystem and JSON utilities
- CLI argument parsing
- Data directory layout for future pipeline stages
- FFmpeg-based conversion from `mp3`, `wav`, `m4a`, or `flac` into preprocessing WAV files
- Silero VAD speech-span detection saved as JSON
- Deterministic segment metadata builder for 3-10 second TTS candidates
- WAV export for final segments with duration validation and an export manifest
- faster-whisper transcription for exported WAV segments, saved as JSONL
- Rule-based Vietnamese-friendly transcript normalization saved as JSONL
- Rule-based quality filtering with accepted/rejected JSONL outputs and a report
- Final TTS dataset export with `metadata.csv`, `manifest.jsonl`, and `summary.json`

Audio copying/moving and advanced dataset packaging are intentionally not implemented yet.

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
|   |-- transcribe.py
|   |-- cleaner.py
|   |-- filter.py
|   `-- exporter.py
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

## Usage

Run the full pipeline:

```bash
python main.py --input data/raw/sample.mp3
```

The command runs preprocessing, VAD, segment metadata building, WAV export, transcription, transcript cleaning, quality filtering, and final dataset metadata export. Preprocessing writes:

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

Use `--output` to override the processed working directory.
