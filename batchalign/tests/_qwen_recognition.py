"""The two Qwen recognition policies the control plane sends today.

They mirror what Rust's ``qwen_recognition`` produces (pinned on the Rust side
by ``a_qwen_worker_spawns_with_its_language_recognition_policy``): Taiwanese
keeps recognizer boundaries, every other language leaves them to segmentation.
Tests construct recognizers with these rather than inventing a third policy.

``FAKE_QWEN_SNAPSHOTS`` stands in for the two resolved snapshots a Qwen load
requires, for tests that replace ``LoadedQwen.load`` and read no weights.
"""

from __future__ import annotations

from batchalign.inference.languages.cantonese._qwen_common import QwenSnapshots
from batchalign.tests._snapshots import transformers_snapshot
from batchalign.worker._types_v2 import (
    BoundedSecondsWindowV2,
    ChunkerDefaultWindowV2,
    QwenRecognitionV2,
    RecognitionPunctuationV2,
)

SEGMENTATION_MODEL_POLICY = QwenRecognitionV2(
    window=ChunkerDefaultWindowV2(),
    punctuation=RecognitionPunctuationV2.DISCARD,
)
"""Every language with a boundary model: chunker window, punctuation discarded."""

TAIWANESE_POLICY = QwenRecognitionV2(
    window=BoundedSecondsWindowV2(seconds=30),
    punctuation=RecognitionPunctuationV2.RESTORE,
)
"""Taiwanese: 30-second windows, recognized punctuation restored."""

FAKE_QWEN_SNAPSHOTS = QwenSnapshots(
    asr=transformers_snapshot("Qwen/Qwen3-ASR-1.7B-hf"),
    aligner=transformers_snapshot("Qwen/Qwen3-ForcedAligner-0.6B-hf"),
)
"""Two snapshots no test reads from: every user replaces ``LoadedQwen.load``."""
