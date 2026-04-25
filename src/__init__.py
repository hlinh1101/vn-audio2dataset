"""Core package for vn-audio2dataset.

The package is intentionally split into small pipeline modules so later steps
can add preprocessing, VAD, segmentation, transcription, cleaning, filtering,
and exporting without turning the CLI into application logic.
"""

__all__ = [
    "config",
    "logger",
    "utils",
]
