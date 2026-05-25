from __future__ import annotations

import unittest

from src.config import AppConfig
from src.stt_workflow import (
    build_speaker_inspection,
    build_timestamp_segments,
    select_target_speaker,
    validate_candidate,
)


def _config() -> AppConfig:
    config = AppConfig()
    config.segments.min_sec = 3.0
    config.segments.max_sec = 15.0
    config.segments.ideal_min = 5.0
    config.segments.ideal_max = 10.0
    config.stt_segmentation.boundary_pad_sec = 0.0
    config.stt_segmentation.boundary_guard_sec = 0.2
    config.stt_segmentation.max_word_gap_sec = 0.8
    config.stt_segmentation.preferred_min_sec = 5.0
    config.stt_segmentation.preferred_max_sec = 10.0
    config.stt_segmentation.semantic_max_sec = 15.0
    config.stt_segmentation.min_boundary_score = 7.0
    config.stt_segmentation.min_words = 3
    config.stt_segmentation.min_speaker_share = 0.6
    return config


def _word(text: str, start: float, end: float, speaker: str) -> dict[str, object]:
    return {
        "text": text,
        "start": start,
        "end": end,
        "type": "word",
        "speaker_id": speaker,
        "logprob": -0.1,
    }


class SttWorkflowTests(unittest.TestCase):
    def test_builds_dominant_speaker_segment(self) -> None:
        response = {
            "language_code": "vi",
            "text": "Xin chao cac ban.",
            "words": [
                _word("Xin", 0.0, 0.7, "speaker_0"),
                _word("chao", 0.8, 1.4, "speaker_0"),
                _word("cac", 1.5, 2.2, "speaker_0"),
                _word("ban.", 2.3, 3.2, "speaker_0"),
            ],
        }

        result = build_timestamp_segments(response, _config())

        self.assertEqual(len(result["segments"]), 1)
        self.assertEqual(result["segments"][0]["speaker_id"], "speaker_0")
        self.assertEqual(result["segments"][0]["text"], "Xin chao cac ban.")

    def test_rejects_mixed_speaker_overlap(self) -> None:
        response = {
            "language_code": "vi",
            "text": "Xin chao chen ngang cac ban.",
            "words": [
                _word("Xin", 0.0, 0.7, "speaker_0"),
                _word("chao", 0.8, 1.4, "speaker_0"),
                _word("chen", 1.45, 1.65, "speaker_1"),
                _word("ngang", 1.7, 1.9, "speaker_1"),
                _word("cac", 2.0, 2.8, "speaker_0"),
                _word("ban.", 2.9, 3.5, "speaker_0"),
            ],
        }

        result = build_timestamp_segments(response, _config())

        self.assertEqual(result["segments"], [])
        self.assertGreaterEqual(len(result["rejected_segments"]), 1)

    def test_does_not_split_inside_vietnamese_named_entity(self) -> None:
        response = {
            "language_code": "vi",
            "text": "Hôm nay đội tuyển quốc gia Việt Nam. Giành chiến thắng.",
            "words": [
                _word("Hôm", 0.0, 0.8, "speaker_0"),
                _word("nay", 0.9, 1.7, "speaker_0"),
                _word("đội", 1.8, 2.6, "speaker_0"),
                _word("tuyển", 2.7, 3.5, "speaker_0"),
                _word("quốc", 3.6, 4.4, "speaker_0"),
                _word("gia", 4.5, 5.3, "speaker_0"),
                _word("Việt", 5.4, 6.2, "speaker_0"),
                _word("Nam.", 6.3, 7.1, "speaker_0"),
                _word("Giành", 7.2, 8.0, "speaker_0"),
                _word("chiến", 8.1, 8.9, "speaker_0"),
                _word("thắng.", 9.0, 9.8, "speaker_0"),
            ],
        }

        result = build_timestamp_segments(response, _config())

        self.assertEqual(len(result["segments"]), 1)
        self.assertIn("Việt Nam.", result["segments"][0]["text"])
        protected_boundaries = [
            row
            for row in result["boundary_scores"]
            if row["left_text"] == "Việt" and row["right_text"] == "Nam."
        ]
        self.assertEqual(len(protected_boundaries), 1)
        self.assertFalse(protected_boundaries[0]["is_valid"])
        self.assertIn("protected_capitalized_entity", protected_boundaries[0]["reasons"])

    def test_accepts_extended_sentence_when_no_earlier_safe_boundary(self) -> None:
        response = {
            "language_code": "vi",
            "text": "Một hai ba bốn năm sáu bảy tám chín mười mười một.",
            "words": [
                _word("Một", 0.0, 0.9, "speaker_0"),
                _word("hai", 1.0, 1.9, "speaker_0"),
                _word("ba", 2.0, 2.9, "speaker_0"),
                _word("bốn", 3.0, 3.9, "speaker_0"),
                _word("năm", 4.0, 4.9, "speaker_0"),
                _word("sáu", 5.0, 5.9, "speaker_0"),
                _word("bảy", 6.0, 6.9, "speaker_0"),
                _word("tám", 7.0, 7.9, "speaker_0"),
                _word("chín", 8.0, 8.9, "speaker_0"),
                _word("mười", 9.0, 9.9, "speaker_0"),
                _word("mười", 10.0, 10.9, "speaker_0"),
                _word("một.", 11.0, 11.9, "speaker_0"),
            ],
        }

        result = build_timestamp_segments(response, _config())

        self.assertEqual(len(result["segments"]), 1)
        self.assertGreater(result["segments"][0]["duration"], 10.0)
        self.assertLessEqual(result["segments"][0]["duration"], 15.0)

    def test_does_not_split_at_comma_even_with_pause(self) -> None:
        response = {
            "language_code": "vi",
            "text": "Mot hai ba bon nam, sau bay tam chin muoi.",
            "words": [
                _word("Mot", 0.0, 0.8, "speaker_0"),
                _word("hai", 0.9, 1.7, "speaker_0"),
                _word("ba", 1.8, 2.6, "speaker_0"),
                _word("bon", 2.7, 3.5, "speaker_0"),
                _word("nam,", 3.6, 4.4, "speaker_0"),
                _word("sau", 5.0, 5.8, "speaker_0"),
                _word("bay", 5.9, 6.7, "speaker_0"),
                _word("tam", 6.8, 7.6, "speaker_0"),
                _word("chin", 7.7, 8.5, "speaker_0"),
                _word("muoi.", 8.6, 9.4, "speaker_0"),
            ],
        }

        result = build_timestamp_segments(response, _config())

        self.assertEqual(len(result["segments"]), 1)
        self.assertEqual(
            result["segments"][0]["text"],
            "Mot hai ba bon nam, sau bay tam chin muoi.",
        )
        comma_boundaries = [
            row for row in result["boundary_scores"] if row["left_text"] == "nam,"
        ]
        self.assertEqual(len(comma_boundaries), 1)
        self.assertFalse(comma_boundaries[0]["is_valid"])
        self.assertIn("not_sentence_boundary", comma_boundaries[0]["reasons"])
        self.assertNotIn("clause_punctuation", comma_boundaries[0]["reasons"])

    def test_accepts_ellipsis_as_sentence_boundary(self) -> None:
        response = {
            "language_code": "vi",
            "text": "Mot hai ba bon...",
            "words": [
                _word("Mot", 0.0, 0.7, "speaker_0"),
                _word("hai", 0.8, 1.5, "speaker_0"),
                _word("ba", 1.6, 2.3, "speaker_0"),
                _word("bon...", 2.4, 3.2, "speaker_0"),
            ],
        }

        result = build_timestamp_segments(response, _config())

        self.assertEqual(len(result["segments"]), 1)
        self.assertIn("sentence_punctuation", result["segments"][0]["boundary_reasons"])

    def test_accepts_unicode_ellipsis_as_sentence_boundary(self) -> None:
        response = {
            "language_code": "vi",
            "text": "Mot hai ba bon…",
            "words": [
                _word("Mot", 0.0, 0.7, "speaker_0"),
                _word("hai", 0.8, 1.5, "speaker_0"),
                _word("ba", 1.6, 2.3, "speaker_0"),
                _word("bon…", 2.4, 3.2, "speaker_0"),
            ],
        }

        result = build_timestamp_segments(response, _config())

        self.assertEqual(len(result["segments"]), 1)
        self.assertIn("sentence_punctuation", result["segments"][0]["boundary_reasons"])

    def test_accepts_overlong_full_sentence_as_single_segment(self) -> None:
        words = [
            _word(f"tu{i}", i * 0.8, i * 0.8 + 0.7, "speaker_0")
            for i in range(22)
        ]
        words[-1]["text"] = f"{words[-1]['text']}."
        response = {
            "language_code": "vi",
            "text": " ".join(str(item["text"]) for item in words),
            "words": words,
        }

        result = build_timestamp_segments(response, _config())

        self.assertEqual(len(result["segments"]), 1)
        self.assertGreater(result["segments"][0]["duration"], 15.0)
        self.assertEqual(result["rejected_segments"], [])

    def test_rejects_overlong_region_without_safe_boundary(self) -> None:
        words = [
            _word(f"từ{i}", i * 0.8, i * 0.8 + 0.7, "speaker_0")
            for i in range(25)
        ]
        response = {
            "language_code": "vi",
            "text": " ".join(str(item["text"]) for item in words),
            "words": words,
        }

        result = build_timestamp_segments(response, _config())

        self.assertEqual(result["segments"], [])
        all_reasons = {
            reason
            for item in result["rejected_segments"]
            for reason in item["reject_reasons"]
        }
        self.assertIn("no_safe_semantic_boundary", all_reasons)

    def test_rejects_audio_event_overlap(self) -> None:
        config = _config()
        words = [
            _word("Xin", 0.0, 0.7, "speaker_0"),
            _word("chao", 0.8, 1.4, "speaker_0"),
            {
                "index": 2,
                "text": "[music]",
                "start": 1.5,
                "end": 1.8,
                "duration": 0.3,
                "type": "audio_event",
                "speaker_id": None,
                "logprob": None,
            },
            _word("cac", 1.9, 2.7, "speaker_0"),
            _word("ban.", 2.8, 3.4, "speaker_0"),
        ]
        normalized = [
            {**item, "index": index, "duration": round(float(item["end"]) - float(item["start"]), 3)}
            for index, item in enumerate(words)
        ]
        candidate = {
            "start": 0.0,
            "end": 3.4,
            "raw_start": 0.0,
            "raw_end": 3.4,
            "duration": 3.4,
            "text": "Xin chao cac ban.",
            "word_count": 4,
            "avg_logprob": -0.1,
        }

        reasons = validate_candidate(candidate, normalized, "speaker_0", config)

        self.assertIn("audio_event_overlap", reasons)

    def test_auto_selection_does_not_fail_on_low_speaker_share(self) -> None:
        config = _config()
        config.stt_segmentation.min_speaker_share = 0.6
        response = {
            "language_code": "vi",
            "text": "Một hai ba bốn. Năm sáu bảy tám. Chín mười mười một.",
            "words": [
                _word("Một", 0.0, 0.9, "speaker_0"),
                _word("hai", 1.0, 1.9, "speaker_0"),
                _word("ba", 2.0, 2.9, "speaker_0"),
                _word("bốn.", 3.0, 4.0, "speaker_0"),
                _word("Năm", 4.2, 5.1, "speaker_1"),
                _word("sáu", 5.2, 6.1, "speaker_1"),
                _word("bảy", 6.2, 7.1, "speaker_1"),
                _word("tám.", 7.2, 8.0, "speaker_1"),
                _word("Chín", 8.2, 9.1, "speaker_2"),
                _word("mười", 9.2, 10.1, "speaker_2"),
                _word("mười", 10.2, 11.1, "speaker_2"),
                _word("một.", 11.2, 12.0, "speaker_2"),
            ],
        }

        result = build_timestamp_segments(response, config)

        self.assertEqual(result["report"]["selected_speaker"], "speaker_0")
        self.assertIn(
            "low_speaker_share",
            result["report"]["speaker_selection"]["selection_warnings"],
        )

    def test_manual_per_file_speaker_selection_exports_configured_speaker(self) -> None:
        config = _config()
        config.stt_segmentation.per_file_target_speakers = {"news_a.wav": "speaker_1"}
        response = {
            "language_code": "vi",
            "text": "Một hai ba bốn. Năm sáu bảy tám.",
            "words": [
                _word("Một", 0.0, 0.9, "speaker_0"),
                _word("hai", 1.0, 1.9, "speaker_0"),
                _word("ba", 2.0, 2.9, "speaker_0"),
                _word("bốn.", 3.0, 4.0, "speaker_0"),
                _word("Năm", 4.2, 5.1, "speaker_1"),
                _word("sáu", 5.2, 6.1, "speaker_1"),
                _word("bảy", 6.2, 7.1, "speaker_1"),
                _word("tám.", 7.2, 8.0, "speaker_1"),
            ],
        }

        result = build_timestamp_segments(response, config, input_path="news_a.wav")

        self.assertEqual(result["report"]["selected_speaker"], "speaker_1")
        self.assertTrue(result["segments"])
        self.assertTrue(
            all(item["speaker_id"] == "speaker_1" for item in result["segments"])
        )

    def test_require_manual_speaker_fails_without_mapping(self) -> None:
        config = _config()
        config.stt_segmentation.speaker_selection_mode = "manual"
        config.stt_segmentation.require_manual_speaker = True
        response = {
            "language_code": "vi",
            "text": "Một hai ba bốn.",
            "words": [
                _word("Một", 0.0, 0.9, "speaker_0"),
                _word("hai", 1.0, 1.9, "speaker_0"),
                _word("ba", 2.0, 2.9, "speaker_0"),
                _word("bốn.", 3.0, 4.0, "speaker_0"),
            ],
        }

        with self.assertRaises(Exception):
            build_timestamp_segments(response, config, input_path="missing.wav")

    def test_speaker_inspection_contains_stats_and_samples(self) -> None:
        response = {
            "language_code": "vi",
            "text": "Một hai ba bốn.",
            "words": [
                _word("Một", 0.0, 0.9, "speaker_0"),
                _word("hai", 1.0, 1.9, "speaker_0"),
                _word("ba", 2.0, 2.9, "speaker_0"),
                _word("bốn.", 3.0, 4.0, "speaker_0"),
            ],
        }

        inspection = build_speaker_inspection(response, _config())

        speaker = inspection["speaker_inspection"]["speakers"]["speaker_0"]
        self.assertEqual(speaker["turn_count"], 1)
        self.assertGreater(speaker["duration_seconds"], 0)
        self.assertTrue(speaker["sample_utterances"])

    def test_select_target_speaker_uses_stem_mapping(self) -> None:
        config = _config()
        config.stt_segmentation.per_file_target_speakers = {"sample": "speaker_1"}
        words = [
            {"index": 0, **_word("Một", 0.0, 1.0, "speaker_0")},
            {"index": 1, **_word("Hai", 1.2, 2.2, "speaker_1")},
        ]

        selected, _stats, selection = select_target_speaker(
            words,
            config,
            input_path="data/raw/sample.mp3",
        )

        self.assertEqual(selected, "speaker_1")
        self.assertEqual(selection["selection_source"], "per_file")


if __name__ == "__main__":
    unittest.main()
