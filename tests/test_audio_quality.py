from __future__ import annotations

import unittest

from src.audio_quality import _apply_quality_rules
from src.config import AppConfig


class AudioQualityTests(unittest.TestCase):
    def test_audio_too_quiet_is_review_not_bad(self) -> None:
        config = AppConfig()
        metrics = {
            "rms_dbfs": config.audio_quality.min_rms_dbfs - 1.0,
            "silence_ratio": 0.0,
            "leading_silence_sec": 0.0,
            "trailing_silence_sec": 0.0,
            "clipping_ratio": 0.0,
            "spectral_flatness": 0.0,
            "high_freq_energy_ratio": 0.0,
        }
        reasons_bad: list[str] = []
        reasons_review: list[str] = []

        _apply_quality_rules(metrics, reasons_bad, reasons_review, config)

        self.assertEqual(reasons_bad, [])
        self.assertIn("audio_too_quiet", reasons_review)


if __name__ == "__main__":
    unittest.main()
