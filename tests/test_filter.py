from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from src.config import AppConfig
from src.filter import filter_record


def _write_silence_wav(path: Path, duration_sec: float = 8.0) -> None:
    sample_rate = 8000
    frame_count = int(sample_rate * duration_sec)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)


class FilterTests(unittest.TestCase):
    def test_long_text_is_not_rejected_by_character_count(self) -> None:
        config = AppConfig()
        config.filter.min_sec = 3.0
        config.filter.max_sec = 15.0
        config.filter.min_words = 3
        config.filter.max_chars = 20
        config.filter.max_unusual_symbol_ratio = None

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "segment.wav"
            _write_silence_wav(audio_path, duration_sec=8.5)
            record = {
                "id": "000001",
                "audio_path": str(audio_path),
                "duration": 8.5,
                "cleaned_text": "Mot cau noi dai hon gioi han ky tu nhung van dung cho TTS.",
            }

            item = filter_record(record, config)

        self.assertTrue(item["accepted"])
        self.assertEqual(item["reject_reasons"], [])
        self.assertGreater(item["char_count"], config.filter.max_chars)
        self.assertGreaterEqual(item["word_count"], config.filter.min_words)

    def test_duration_remains_primary_length_rejection(self) -> None:
        config = AppConfig()
        config.filter.min_sec = 3.0
        config.filter.max_sec = 15.0
        config.filter.min_words = 3
        config.filter.max_chars = 20
        config.filter.max_unusual_symbol_ratio = None

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "segment.wav"
            _write_silence_wav(audio_path, duration_sec=16.0)
            record = {
                "id": "000002",
                "audio_path": str(audio_path),
                "duration": 16.0,
                "cleaned_text": "Mot cau noi dai hon gioi han ky tu nhung qua dai theo thoi luong.",
            }

            item = filter_record(record, config)

        self.assertFalse(item["accepted"])
        self.assertIn("duration_too_long", item["reject_reasons"])
        self.assertNotIn("char_count_too_high", item["reject_reasons"])
        self.assertGreater(item["char_count"], config.filter.max_chars)


if __name__ == "__main__":
    unittest.main()
