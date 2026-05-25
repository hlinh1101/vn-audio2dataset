# Workflow va Source Map du an `vn-audio2dataset`

Tai lieu nay tom tat workflow thuc te cua du an dua tren `main.py`, `config.yaml`, `README.md` va cac module trong `src/`. Muc tieu cua project la bien mot file audio dai tieng Viet thanh dataset TTS gom cac file WAV ngan va `metadata.csv`.

## 1. Tong quan pipeline

Du an co hai cach tao dataset chinh:

1. **Pipeline VAD truyen thong** (`--stage full`, mac dinh): cat audio bang Silero VAD truoc, sau do moi transcribe tung doan.
2. **Pipeline transcript-first** (`--stage stt-full`): gui ca file audio da tien xu ly len ElevenLabs Scribe v2 de lay word timestamps va diarization, sau do cat audio theo transcript/speaker.

Ngoai ra co cac stage doc lap de chay lai tung phan: `clean`, `quality`, `filter`, `export`, `consolidate`, va stage `stt-inspect` de kiem tra speaker truoc khi export dataset.

Thu muc chinh:

```text
D:\KhoaLuan
|-- main.py                 # CLI entrypoint va dieu phoi pipeline
|-- config.yaml             # Cau hinh threshold, backend, path, speaker
|-- src/                    # Logic pipeline
|-- scripts/                # Utility ngoai pipeline chinh
|-- tests/                  # Unit tests
`-- data/
    |-- raw/                # Audio goc
    |-- processed/          # Audio da preprocess
    |-- output/             # Artifact va dataset dau ra
    `-- rejects/            # Du phong cho file bi loai
```

## 2. CLI va cac stage trong `main.py`

`main.py` la entrypoint duy nhat cua workflow. Ham `parse_args()` doc tham so CLI, `load_config()` nap `config.yaml`, `setup_logger()` cau hinh log, sau do `main()` dispatch theo `--stage`.

| Stage | Lenh mau | Chuc nang |
| --- | --- | --- |
| `full` | `python main.py --input data/raw/sample.wav` | Chay pipeline VAD day du tu audio goc den dataset. Day la stage mac dinh. |
| `stt-full` | `python main.py --stage stt-full --input data/raw/sample.wav` | Chay workflow ElevenLabs transcript-first, co word timestamps va speaker diarization. |
| `stt-inspect` | `python main.py --stage stt-inspect --input data/raw/sample.wav` | Chi tao artifact kiem tra speaker, dung truoc segmentation/export. |
| `clean` | `python main.py --stage clean --raw data/output/sample/raw_transcripts.jsonl` | Lam sach transcript co san. |
| `quality` | `python main.py --stage quality --manifest data/output/sample/export_manifest.json` | Phan loai audio da cat thanh good/review/bad. |
| `filter` | `python main.py --stage filter --cleaned data/output/sample/cleaned_transcripts.jsonl` | Loc transcript/audio thanh accepted/rejected. |
| `export` | `python main.py --stage export --accepted data/output/sample/accepted.jsonl` | Tao `metadata.csv`, `manifest.jsonl`, `summary.json`. |
| `consolidate` | `python main.py --stage consolidate --source ... --output ...` | Gop nhieu output folder dua tren `accepted.jsonl`. |

Rang buoc CLI quan trong:

- `--input-dir` chi dung voi `stt-full` va `stt-inspect`.
- `--stt-include-review` chi dung voi `stt-full`; no dua ca clip `audio_quality_review.json` vao `raw_transcripts.jsonl`, nhung van loai `audio_quality_bad.json`.
- `--force` chi dung voi `stt-full` va `stt-inspect`; no bo qua cache `elevenlabs_full_transcript.json` va khong skip output da hoan tat trong batch `stt-full`.
- `consolidate` bat buoc co it nhat mot `--source` va co `--output`.

## 3. Workflow `full`: VAD truoc, transcribe sau

Lenh mac dinh:

```powershell
python main.py --input data/raw/sample.wav
```

Thu tu code trong `run_full_pipeline()`:

1. `preprocess_audio()` tao audio lam viec:
   - `data/processed/<stem>/audio_16k.wav`
   - `data/processed/<stem>/audio_master.wav`
2. `run_vad()` chay Silero VAD tren `audio_16k.wav`.
3. `save_vad_segments()` ghi `data/output/<stem>/vad_segments.json`.
4. `load_vad_segments()` va `build_segments()` gom/tach speech spans thanh segment ung vien.
5. `save_final_segments()` ghi `final_segments.json`.
6. `cut_audio_segments()` cat `audio_master.wav` thanh `wavs/*.wav`.
7. Ghi `export_manifest.json`.
8. `filter_audio_manifest()` phan loai audio:
   - `audio_quality_good.json`
   - `audio_quality_review.json`
   - `audio_quality_bad.json`
   - `audio_quality_report.json`
9. Chi `audio_quality_good.json` duoc dua vao `transcribe_all_segments()`.
10. `raw_transcripts.jsonl` duoc ghi tang dan trong luc transcribe.
11. `clean_all_transcripts()` tao `cleaned_transcripts.jsonl`.
12. `filter_all()` tao:
   - `accepted.jsonl`
   - `rejected.jsonl`
   - `filter_report.json`
13. `export_dataset_from_accepted()` tao dataset cuoi:
   - `metadata.csv`
   - `manifest.jsonl`
   - `summary.json`

Output folder mac dinh la `data/output/<stem>/`.

## 4. Workflow `stt-full`: transcript-first voi ElevenLabs

Lenh mot file:

```powershell
python main.py --stage stt-full --input data/raw/sample.wav
```

Lenh batch:

```powershell
python main.py --stage stt-full --input-dir data/raw/mctuanduong
```

Thu tu code trong `run_stt_full_file()`:

1. `preprocess_audio()` van tao `audio_16k.wav` va `audio_master.wav`.
2. `_load_or_create_full_stt_response()` doc cache `data/output/<stem>_stt/elevenlabs_full_transcript.json` neu co; neu khong co hoac co `--force` thi goi ElevenLabs.
3. `_save_stt_speaker_inspection()` tao artifact kiem tra speaker:
   - `stt_words.json`
   - `stt_speaker_inspection.json`
   - `stt_speaker_turns.json`
4. `build_timestamp_segments()` trong `src/stt_workflow.py` chon target speaker va tao segment bang word timestamps.
5. Ghi artifact segmentation:
   - `stt_segments.json`
   - `stt_rejected_segments.json`
   - `stt_boundary_scores.json`
   - `stt_segmentation_report.json`
6. `cut_audio_segments()` cat tu `audio_master.wav` vao `wavs/`.
7. `filter_audio_manifest()` chay audio quality gate.
8. `write_transcripts_from_stt_segments()` tao `raw_transcripts.jsonl` tu transcript da co, khong transcribe lai tung clip.
9. Chay lai cac buoc chung: clean -> filter -> export dataset.

Khac biet quan trong so voi `full`:

- `stt-full` transcribe ca file truoc, cat audio sau.
- Segment boundary uu tien dau cau ket thuc nhu `.`, `?`, `!`, `...`, khong cat theo dau phay.
- Co speaker selection dua vao `config.yaml`, dac biet cac key `speaker_selection_mode`, `default_target_speaker`, `require_manual_speaker`, `per_file_target_speakers`.
- Output folder mac dinh la `data/output/<stem>_stt/`.
- Trong batch, mot file chi bi skip khi output folder da co du `summary.json`, `metadata.csv`, `manifest.jsonl`, va `accepted.jsonl`.

## 5. Workflow `stt-inspect`: kiem tra speaker truoc

Lenh:

```powershell
python main.py --stage stt-inspect --input data/raw/sample.wav
python main.py --stage stt-inspect --input-dir data/raw/mctuanduong
```

Stage nay dung chung cache full transcript voi `stt-full`, sau do ghi:

- `elevenlabs_full_transcript.json`
- `stt_words.json`
- `stt_speaker_inspection.json`
- `stt_speaker_turns.json`

No dung lai sau buoc inspection, khong tao `stt_segments.json`, khong cat WAV, khong clean/filter/export dataset. Day la mode dung de xem file co bao nhieu speaker, speaker nao co duration/share/word count phu hop, roi cap nhat `per_file_target_speakers` trong `config.yaml`.

## 6. Audio quality gate

`src/audio_quality.py` doc `export_manifest.json`, phan tich tung WAV va gan nhan:

- `good`: du chat luong, duoc di tiep.
- `review`: can xem lai; mac dinh khong di tiep trong `full`, nhung co the di tiep trong `stt-full` neu dung `--stt-include-review`.
- `bad`: bi loai.

Metric chinh:

| Metric | Y nghia |
| --- | --- |
| `duration` | Do dai clip; qua ngan/qua dai bi loai. |
| `rms_dbfs` | Am luong trung binh; qua nho dua vao review. |
| `clipping_ratio` | Ti le mau gan tran bien do; cao thi nghi clipping. |
| `silence_ratio` | Ti le im lang noi bo. |
| `leading_silence_sec` | Im lang dau clip. |
| `trailing_silence_sec` | Im lang cuoi clip. |
| `spectral_flatness` | Proxy cho nhieu/reverb. |
| `high_freq_energy_ratio` | Proxy cho nhieu tan so cao. |

Nguong hien tai nam trong block `audio_quality:` cua `config.yaml`.

## 7. Dataset export va consolidate

`src/exporter.py` tao dataset tu `accepted.jsonl`:

- Cat/validate audio segment khi can.
- Ghi `metadata.csv` theo format TTS `filename.wav|text`.
- Ghi `manifest.jsonl` co metadata day du hon.
- Ghi `summary.json` gom tong so mau va tong duration.

`src/consolidator.py` gop nhieu output folder da hoan tat:

- Moi source phai co `accepted.jsonl`.
- Audio trong `accepted.jsonl` duoc copy sang output moi.
- Output moi duoc regenerate tu accepted records:
  - `accepted.jsonl`
  - `metadata.csv`
  - `manifest.jsonl`
  - `summary.json`
  - `consolidation_report.json`

Luu y: `consolidate` **khong doc source `metadata.csv`**. Neu can bao toan cac dong `metadata.csv` da sua tay, dung script `scripts/merge_metadata_csv_dataset.py`.

## 8. Chuc nang tung file Python trong `src/`

| File | Vai tro |
| --- | --- |
| `src/__init__.py` | Danh dau `src` la package Python va khai bao metadata package neu co. |
| `src/config.py` | Dinh nghia cac dataclass cau hinh (`AudioConfig`, `SegmentConfig`, `VadConfig`, `TranscriptionConfig`, `AudioQualityConfig`, `SttSegmentationConfig`, `FilterConfig`, `PathConfig`, `LoggingConfig`, `AppConfig`) va ham `load_config()` de nap `config.yaml`. File nay chuyen YAML thanh object co type ro rang cho toan pipeline. |
| `src/logger.py` | Cung cap `setup_logger()` de tao logger dung chung theo project name, level va optional log file. |
| `src/utils.py` | Cac helper nho dung chung: tao thu muc (`ensure_dir`), tao timestamp, lam sach stem filename (`safe_stem`), doc/ghi JSON (`load_json`, `save_json`). |
| `src/preprocess.py` | Xu ly audio dau vao. Kiem tra dinh dang ho tro, goi FFmpeg de tao `audio_16k.wav` cho VAD/STT va `audio_master.wav` cho cat segment/export. |
| `src/vad.py` | Nap Silero VAD, kiem tra WAV 16 kHz mono PCM, chay VAD de lay speech spans, chuan hoa segment va ghi `vad_segments.json`. |
| `src/segmenter.py` | Doc `vad_segments.json`, gom cac speech spans ngan, tach segment qua dai, tao segment cuoi phu hop nguong trong `segments:` cua config, va ghi `final_segments.json`. |
| `src/audio_quality.py` | Doc `export_manifest.json`, phan tich WAV bang `soundfile`/`numpy`, gan nhan `good`/`review`/`bad`, ghi cac manifest chat luong va `audio_quality_report.json`. |
| `src/transcribe.py` | Transcribe cac WAV segment bang backend `faster_whisper` hoac `elevenlabs`. Ho tro doc `.env`, retry ElevenLabs, ghi JSONL tang dan, reuse transcript da co trong `raw_transcripts.jsonl`, va giu cache model faster-whisper de tranh loi teardown tren Windows/CUDA. |
| `src/stt_workflow.py` | Workflow transcript-first. Goi ElevenLabs full-audio STT, luu response, chuan hoa word timestamps, thong ke speaker, chon target speaker, tao speaker inspection, cat semantic segments dua tren transcript, bao ve boundary rui ro, va tao `raw_transcripts.jsonl` tu STT segments. |
| `src/cleaner.py` | Lam sach transcript JSONL: normalize Unicode/text, loai emoji neu config bat, xu ly quote/lowercase theo config, ghi `cleaned_transcripts.jsonl` va thong ke row rong sau cleaning. |
| `src/filter.py` | Danh gia chat luong record sau cleaning. Kiem tra duration, word count, RMS optional, no-speech/logprob optional, unusual symbol ratio, va tach thanh `accepted.jsonl`, `rejected.jsonl`, `filter_report.json`. `char_count` duoc giu trong metadata/diagnostic, khong phai gate chinh khi config hien tai dat char-count la thong tin tham khao. |
| `src/exporter.py` | Doc accepted records, validate/cut audio segment, tao `metadata.csv`, `manifest.jsonl`, `summary.json`. Day la buoc bien artifact pipeline thanh dataset TTS cuoi cung. |
| `src/consolidator.py` | Gop nhieu output folder dua tren `accepted.jsonl`. Validate source/target, copy audio voi ten an toan/khong trung, regenerate dataset output va `consolidation_report.json`. |

## 9. Utility ngoai `src/`

| File | Vai tro |
| --- | --- |
| `scripts/merge_metadata_csv_dataset.py` | Gop cac folder `mc*_stt` bang cach doc truc tiep `metadata.csv` va copy file trong `wavs/`. Script nay tao output toi gian chi gom `metadata.csv` va `wavs/`, phu hop khi metadata da duoc sua tay va can giu nguyen noi dung text. |

Lenh mac dinh cua script:

```powershell
python scripts/merge_metadata_csv_dataset.py --source-root data/output --start 1 --end 31 --output data/output/datasets/mc1_mc31_metadata
```

Them `--force` neu output folder da ton tai va muon tao lai.

## 10. Artifact chinh theo workflow

| Artifact | Tao boi | Y nghia |
| --- | --- | --- |
| `audio_16k.wav` | `preprocess.py` | Audio 16 kHz mono cho VAD hoac full-audio STT. |
| `audio_master.wav` | `preprocess.py` | Audio chat luong cao hon de cat segment WAV. |
| `vad_segments.json` | `vad.py` | Speech spans tu Silero VAD. |
| `final_segments.json` | `segmenter.py` | Segment cuoi cho workflow VAD. |
| `elevenlabs_full_transcript.json` | `stt_workflow.py` | Cache response STT full-audio cua ElevenLabs. |
| `stt_speaker_inspection.json` | `stt_workflow.py` | Bao cao speaker de review/chon target speaker. |
| `stt_segments.json` | `stt_workflow.py` | Segment accepted cua workflow transcript-first. |
| `stt_rejected_segments.json` | `stt_workflow.py` | Segment bi loai va ly do trong workflow transcript-first. |
| `export_manifest.json` | `exporter.py` / `main.py` | Danh sach WAV da cat truoc audio quality. |
| `audio_quality_good.json` | `audio_quality.py` | Clip dat chat luong de di tiep. |
| `audio_quality_review.json` | `audio_quality.py` | Clip bien gioi, can review. |
| `audio_quality_bad.json` | `audio_quality.py` | Clip bi loai. |
| `raw_transcripts.jsonl` | `transcribe.py` / `stt_workflow.py` | Transcript raw cho tung clip. |
| `cleaned_transcripts.jsonl` | `cleaner.py` | Transcript da normalize/clean. |
| `accepted.jsonl` | `filter.py` | Record duoc chap nhan de export dataset. |
| `rejected.jsonl` | `filter.py` | Record bi loai va ly do. |
| `filter_report.json` | `filter.py` | Bao cao thong ke filter. |
| `metadata.csv` | `exporter.py` | File metadata TTS `filename.wav|text`. |
| `manifest.jsonl` | `exporter.py` | Manifest dataset chi tiet hon metadata.csv. |
| `summary.json` | `exporter.py` | Tong ket so mau va duration dataset. |
| `consolidation_report.json` | `consolidator.py` | Bao cao merge nhieu output folder. |

## 11. Ghi chu cau hinh quan trong

`config.yaml` dieu khien hanh vi pipeline:

- `transcription_backend`: chon `faster_whisper` hoac `elevenlabs`.
- `audio`: sample rate, channel, normalize.
- `segments`: nguong segment cho workflow VAD.
- `vad`: tham so Silero VAD.
- `transcription`: model, language, beam size, device.
- `cleaning`: lowercase, emoji, quote.
- `audio_quality`: nguong good/review/bad.
- `stt_segmentation`: model ElevenLabs, diarization, speaker selection, boundary scoring.
- `filter`: nguong loc sau cleaning.
- `paths`: `data/raw`, `data/processed`, `data/output`, `data/rejects`.
- `logging`: level va optional log file.

Voi du lieu broadcast/news nhieu speaker, cac key quan trong nhat la:

- `stt_segmentation.speaker_selection_mode`
- `stt_segmentation.require_manual_speaker`
- `stt_segmentation.default_target_speaker`
- `stt_segmentation.per_file_target_speakers`
- `stt_segmentation.min_speaker_share`

## 12. Cach doc source nhanh

Neu can debug workflow, nen bat dau theo thu tu:

1. `main.py`: xem stage nao goi ham nao va artifact duoc ghi o dau.
2. `config.yaml`: xem threshold/backend/path hien tai.
3. Module trong `src/` tuong ung voi artifact dang loi.
4. `tests/`: xem hanh vi nao da co unit test, dac biet `test_stt_workflow.py`, `test_audio_quality.py`, `test_filter.py`, `test_merge_metadata_csv_dataset.py`.

Mapping loi thuong gap:

| Trieu chung | Nen doc file |
| --- | --- |
| Loi FFmpeg/preprocess | `src/preprocess.py` |
| VAD khong ra segment | `src/vad.py`, `src/segmenter.py` |
| Segment STT bi cat sai cau/speaker | `src/stt_workflow.py`, `config.yaml` |
| Clip bi loai good/review/bad | `src/audio_quality.py`, `config.yaml` |
| Transcribe dung sau `raw_transcripts.jsonl` | `src/transcribe.py`, `main.py` |
| Transcript bi clean rong | `src/cleaner.py` |
| Accepted/rejected khong nhu mong doi | `src/filter.py`, `filter_report.json` |
| Metadata/export sai | `src/exporter.py` |
| Merge dataset khong giu metadata da sua | `src/consolidator.py`, `scripts/merge_metadata_csv_dataset.py` |
