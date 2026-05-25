from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from scripts.merge_metadata_csv_dataset import (
    MetadataMergeError,
    merge_metadata_dataset,
    parse_metadata_line,
)


def _write_silence_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b"\x00\x00" * 800)


class MetadataCsvMergeTests(unittest.TestCase):
    def test_parse_metadata_line_keeps_text_after_first_separator(self) -> None:
        row = parse_metadata_line(
            "000001.wav|Xin chao | van giu dau gach doc",
            Path("metadata.csv"),
            1,
        )

        self.assertEqual(row.audio_name, "000001.wav")
        self.assertEqual(row.text, "Xin chao | van giu dau gach doc")

    def test_parse_metadata_line_rejects_missing_separator(self) -> None:
        with self.assertRaises(MetadataMergeError):
            parse_metadata_line("000001.wav Xin chao", Path("metadata.csv"), 1)

    def test_merge_metadata_dataset_copies_referenced_wavs_and_prefixes_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "output"
            for index, text in [(1, "Cau thu nhat"), (2, "Cau thu hai")]:
                source_dir = root / f"mc{index}_stt"
                wavs_dir = source_dir / "wavs"
                wavs_dir.mkdir(parents=True)
                _write_silence_wav(wavs_dir / "000001.wav")
                (source_dir / "metadata.csv").write_text(
                    f"000001.wav|{text}\n",
                    encoding="utf-8",
                    newline="\n",
                )

            output_dir = Path(temp_dir) / "merged"
            stats = merge_metadata_dataset(root, 1, 2, output_dir)

            self.assertEqual(stats.source_count, 2)
            self.assertEqual(stats.row_count, 2)
            self.assertEqual(stats.wav_count, 2)
            self.assertTrue((output_dir / "wavs" / "mc1_stt_000001.wav").exists())
            self.assertTrue((output_dir / "wavs" / "mc2_stt_000001.wav").exists())
            self.assertEqual(
                (output_dir / "metadata.csv").read_text(encoding="utf-8").splitlines(),
                [
                    "mc1_stt_000001.wav|Cau thu nhat",
                    "mc2_stt_000001.wav|Cau thu hai",
                ],
            )


if __name__ == "__main__":
    unittest.main()
