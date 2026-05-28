"""Regression test for the Qwen3-ASR forced-aligner wiring contract.

Origin: 2026-05-27 v2 Cantonese ASR benchmark Bucket A.
``QwenRecognizer`` called ``Qwen3ASRModel.from_pretrained(...)`` without
a ``forced_aligner`` argument, then asked for ``return_time_stamps=True``
at transcribe time. The qwen-asr library rejects that combination
unconditionally (``qwen_asr/inference/qwen3_asr.py:336``):

    ValueError: return_time_stamps=True requires `forced_aligner`
    to be provided at initialization.

Symptom in production: the worker raised the ValueError inside
``infer_qwen_asr_v2``; the exception did not surface to the Rust side
through the V2 IPC, so ``batchalign3 benchmark --engine-overrides
'{"asr":"qwen",...}' --sequential --no-server`` deadlocked at 0% CPU
for 32+ minutes before the operator killed it.

Principled fix: pass ``forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B"``
to ``from_pretrained`` (the aligner model qwen-asr's own README
documents as the canonical pairing). Word-level timestamps are
load-bearing for downstream FA tier injection; the alternative —
``return_time_stamps=False`` — would silently drop word timing and
degrade the entire qwen-asr pipeline.

The unit test pins the wire contract via monkeypatch and runs fast.
The integration test exercises the real model on real audio and is
gated behind ``BATCHALIGN_QWEN_INTEGRATION_AUDIO`` so CI without the
~2 GB Qwen3-ASR-0.6B weights and the audio fixture skips cleanly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _integration_audio() -> Path | None:
    """Return the integration-test audio path if the operator pointed at one."""
    env_path = os.environ.get("BATCHALIGN_QWEN_INTEGRATION_AUDIO")
    if not env_path:
        return None
    candidate = Path(env_path)
    return candidate if candidate.is_file() else None


@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
def test_qwen_recognizer_warm_passes_forced_aligner_to_from_pretrained(monkeypatch) -> None:
    """``QwenRecognizer.warm()`` must pass a non-empty ``forced_aligner``
    argument to ``Qwen3ASRModel.from_pretrained``.

    The 2026-05-27 bug omitted this argument while still asking
    ``model.transcribe(..., return_time_stamps=True)`` — an
    unconditional ValueError from qwen-asr. This test catches both the
    original regression and any future "shortcut" fix that flips
    ``return_time_stamps`` to ``False`` on the transcribe call.
    """
    captured_kwargs: dict[str, object] = {}

    class _SpyModel:
        """Records ``from_pretrained`` kwargs without loading the real model."""

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):  # noqa: ANN001 — match library signature
            captured_kwargs["model_id"] = model_id
            captured_kwargs.update(kwargs)
            return cls()

    monkeypatch.setattr(
        "qwen_asr.Qwen3ASRModel",
        _SpyModel,
    )

    from batchalign.inference.languages.cantonese._qwen_common import QwenRecognizer

    recognizer = QwenRecognizer(
        lang="yue",
        model_id="Qwen/Qwen3-ASR-0.6B",
        device="cpu",
    )
    recognizer.warm()

    forced_aligner = captured_kwargs.get("forced_aligner")
    assert forced_aligner, (
        "Qwen3ASRModel.from_pretrained was called without `forced_aligner`. "
        "qwen-asr requires this argument when downstream uses "
        "return_time_stamps=True (qwen3_asr.py:336). Word-level "
        "timestamps are load-bearing for FA tier injection — do not "
        "remove return_time_stamps=True from transcribe() as a shortcut."
    )
    assert isinstance(forced_aligner, str), (
        "`forced_aligner` should be a HuggingFace repo id string; "
        f"got {type(forced_aligner).__name__}"
    )

    forced_aligner_kwargs = captured_kwargs.get("forced_aligner_kwargs")
    assert isinstance(forced_aligner_kwargs, dict) and forced_aligner_kwargs, (
        "`forced_aligner_kwargs` must propagate device + dtype so the "
        "aligner model loads on the same device as the ASR model. "
        f"got {forced_aligner_kwargs!r}"
    )


@pytest.mark.integration
@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
@pytest.mark.skipif(
    _integration_audio() is None,
    reason="set BATCHALIGN_QWEN_INTEGRATION_AUDIO to a real Cantonese .wav to run",
)
def test_qwen_recognizer_transcribe_produces_word_timestamps() -> None:
    """End-to-end: real Qwen3-ASR-0.6B + Qwen3-ForcedAligner-0.6B on a
    real Cantonese audio fixture must return non-empty word-level
    timing.

    Skipped unless the operator points ``BATCHALIGN_QWEN_INTEGRATION_AUDIO``
    at a Cantonese .wav file. Expected first-run cost: ~2 GB model
    download (cached) plus ~1-3 min CPU inference per minute of audio.
    Locally run via::

        BATCHALIGN_QWEN_INTEGRATION_AUDIO=/path/to/cantonese.wav \\
        uv run pytest -m integration -k qwen_recognizer_transcribe -v
    """
    audio = _integration_audio()
    assert audio is not None  # narrows for type checker; skipif already guards

    from batchalign.inference.languages.cantonese._qwen_common import QwenRecognizer

    recognizer = QwenRecognizer(
        lang="yue",
        model_id="Qwen/Qwen3-ASR-0.6B",
        device="cpu",
    )
    recognizer.warm()
    payload, timed_words = recognizer.transcribe(str(audio))

    assert timed_words, (
        "QwenRecognizer.transcribe returned zero timed words on real "
        "Cantonese audio. Either the aligner is not wired (no word "
        "timestamps emitted) or the audio is silent. Inspect the "
        "raw qwen-asr output before changing this assertion."
    )
    for tw in timed_words:
        start_ms = tw["start_ms"]
        end_ms = tw["end_ms"]
        assert start_ms >= 0, f"negative start_ms in TimedWord: {tw!r}"
        assert end_ms >= start_ms, (
            f"end_ms < start_ms in TimedWord: {tw!r} — aligner output "
            f"violates timing monotonicity"
        )

    assert payload["monologues"], "transcribe payload has no monologues"
