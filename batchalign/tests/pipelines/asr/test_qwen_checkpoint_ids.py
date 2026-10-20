"""Checkpoint-id handling for the native Qwen3-ASR engine.

The sidecar's bare repository ids (``Qwen/Qwen3-ASR-0.6B``) and the native
module's ``-hf`` ids both exist on the Hub, so a stale
``--engine-overrides '{"qwen_model": ...}'`` invocation would otherwise fail
somewhere deep inside a model loader instead of saying what to change. The
bare spelling was published in this project's own documentation for months,
so this path is reached by real command lines that already exist, not just
hypothetically.
"""

from __future__ import annotations

import pytest

from batchalign.inference.languages.cantonese._qwen_common import QwenRecognizerConfig
from batchalign.tests._qwen_recognition import SEGMENTATION_MODEL_POLICY


def test_a_pre_migration_checkpoint_id_is_refused_with_an_instruction() -> None:
    with pytest.raises(ValueError, match=r"Qwen/Qwen3-ASR-0\.6B-hf"):
        QwenRecognizerConfig(
            lang="yue",
            model_id="Qwen/Qwen3-ASR-0.6B",
            device="cpu",
            recognition=SEGMENTATION_MODEL_POLICY,
        )


def test_the_hf_checkpoint_id_is_accepted() -> None:
    recognizer = QwenRecognizerConfig(
        lang="yue",
        model_id="Qwen/Qwen3-ASR-0.6B-hf",
        device="cpu",
        recognition=SEGMENTATION_MODEL_POLICY,
    )
    assert recognizer.model_id == "Qwen/Qwen3-ASR-0.6B-hf"


def test_a_non_qwen_checkpoint_is_left_alone() -> None:
    """The guard is about ONE family's rename, not a general spelling rule."""
    recognizer = QwenRecognizerConfig(
        lang="yue",
        model_id="some-org/some-other-asr",
        device="cpu",
        recognition=SEGMENTATION_MODEL_POLICY,
    )
    assert recognizer.model_id == "some-org/some-other-asr"
