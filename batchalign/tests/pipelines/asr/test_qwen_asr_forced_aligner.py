"""End-to-end check that the Qwen3-ASR engine emits word-level timing.

Origin: 2026-05-27 v2 Cantonese ASR benchmark Bucket A. The recognizer loaded
an ASR model without a forced aligner and then asked for timestamps anyway.
The sidecar rejected that combination, the exception did not surface through
the V2 IPC, and a benchmark run sat at 0% CPU for 32 minutes before the
operator killed it. Word-level timestamps are load-bearing: the FA pipeline
injects them into `%wor`.

**The unit test that used to live here is gone, and its absence is the point.**
It monkeypatched the loader and asserted that a `forced_aligner` argument was
passed, which pinned a CALL SHAPE rather than an outcome: it could only ever
catch the mistake after someone had already written it, and it broke the moment
the engine moved off that package. The engine now loads through `LoadedQwen`,
a type with one constructor that produces both models or raises, so a
recognizer holding an ASR model and no aligner cannot be built at all. That
deleted the defect rather than watching for it, and the test went with it
(2026-08-20, native-transformers migration).

What survives here is what no type of ours can reach: real weights, real audio,
and whether the numbers that come back are usable. That still needs a test.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from batchalign.tests._qwen_recognition import SEGMENTATION_MODEL_POLICY


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


@pytest.mark.integration
@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
@pytest.mark.skipif(
    _integration_audio() is None,
    reason="set BATCHALIGN_QWEN_INTEGRATION_AUDIO to a real Cantonese .wav to run",
)
def test_qwen_recognizer_transcribe_produces_word_timestamps() -> None:
    """Real Qwen3-ASR-0.6B-hf + Qwen3-ForcedAligner-0.6B-hf on real audio must
    return non-empty, well-ordered, recording-relative word timing.

    Skipped unless the operator points ``BATCHALIGN_QWEN_INTEGRATION_AUDIO``
    at a Cantonese .wav file. Expected first-run cost: ~2 GB model download
    (cached) plus roughly real-time CPU inference. Locally run via::

        BATCHALIGN_QWEN_INTEGRATION_AUDIO=/path/to/cantonese.wav \\
        uv run pytest -m integration -k qwen_recognizer_transcribe -v
    """
    audio = _integration_audio()
    assert audio is not None  # narrows for type checker; skipif already guards

    from batchalign.inference.languages.cantonese._qwen_common import (
        QwenRecognizerConfig,
    )

    config = QwenRecognizerConfig(
        lang="yue",
        model_id="Qwen/Qwen3-ASR-0.6B-hf",
        device="cpu",
        recognition=SEGMENTATION_MODEL_POLICY,
    )
    recognizer = config.load()
    _payload, timed_words = recognizer.transcribe(str(audio))

    assert timed_words, (
        "ReadyQwenRecognizer.transcribe returned zero timed words on real "
        "Cantonese audio. Either the aligner produced nothing or the audio "
        "is silent. Inspect the raw output before changing this assertion."
    )
    for tw in timed_words:
        start_ms = tw["start_ms"]
        end_ms = tw["end_ms"]
        assert start_ms >= 0, f"negative start_ms in TimedWord: {tw!r}"
        assert end_ms >= start_ms, (
            f"end_ms < start_ms in TimedWord: {tw!r}, aligner output "
            f"violates timing monotonicity"
        )

    # Recording-relative, not window-relative. Under chunking the aligner
    # reports against the 180 s window it saw, so a missing offset conversion
    # leaves every word after the first chunk stacked at the start of the
    # file. A transcript whose words all fall inside the first window, on
    # audio longer than one window, is that bug.
    import librosa

    duration = float(librosa.get_duration(path=str(audio)))
    last_end_s = max(tw["end_ms"] for tw in timed_words) / 1000.0
    if duration > 200.0:
        assert last_end_s > 180.0, (
            f"audio is {duration:.0f}s but no word ends after 180s "
            f"(last ends at {last_end_s:.1f}s): word timings look "
            f"window-relative, i.e. the chunk offset was not applied."
        )
    assert last_end_s <= duration + 1.0, (
        f"a word ends at {last_end_s:.1f}s, past the {duration:.1f}s end of "
        f"its own audio: chunk offsets are being applied twice or wrongly."
    )
