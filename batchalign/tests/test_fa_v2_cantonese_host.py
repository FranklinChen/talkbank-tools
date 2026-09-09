# affects: batchalign/worker/_fa_v2.py
"""The V2 FA execution host must be able to serve the Cantonese engine.

The host has carried a `canto_runner` field and the Rust bridge a matching
dispatch arm, while nothing built the runner, so a Cantonese request could not
be served even once the routing constructed one. The jyutping conversion was
unit-tested throughout; what was missing was the assembly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from batchalign.worker._fa_v2 import (
    ForcedAlignmentRequestV2,
    PreparedFaPayloadV2,
    build_default_fa_execution_host_v2,
)


class _FakeRomanizer:
    """Stands in for pycantonese, recording what it was asked to convert."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def characters_to_jyutping(self, word: str) -> list[tuple[str, str]]:
        self.seen.append(word)
        return [(word, f"jyut-{word}")]


def _host_with_canto() -> tuple[Any, list[list[str]]]:
    """A Cantonese host whose aligner records the words it actually received."""
    from batchalign.inference.languages.cantonese._cantonese_fa import CantoneseFaHost

    aligned: list[list[str]] = []

    def _fake_align(_model: object, _audio: object, words: list[str]):
        aligned.append(list(words))
        return [(w, (i * 100, i * 100 + 90), None) for i, w in enumerate(words)]

    host = CantoneseFaHost(
        model=object(),
        romanizer=_FakeRomanizer(),
        load_audio_file=lambda _path: None,  # unused on the V2 path
        infer_wave2vec_fa=_fake_align,
    )
    return host, aligned


def test_no_canto_host_means_no_canto_runner() -> None:
    """Absence stays absent; the bridge reports ModelUnavailable, as before."""
    host = build_default_fa_execution_host_v2(
        whisper_model=None, wave2vec_model=None, canto_host=None, qwen_host=None
    )
    assert host.canto_runner is None


def test_a_canto_host_produces_a_runner() -> None:
    """The gap this test exists for: a loaded model must reach the bridge."""
    canto_host, _ = _host_with_canto()
    host = build_default_fa_execution_host_v2(
        whisper_model=None,
        wave2vec_model=None,
        canto_host=canto_host,
        qwen_host=None,
    )
    assert host.canto_runner is not None


def test_the_runner_romanizes_before_aligning() -> None:
    """Jyutping conversion is the whole reason this engine exists.

    Aligning Han characters directly against a wav2vec model trained on
    romanized Cantonese is what selecting `wav2vec_canto` is meant to avoid,
    and is what the collapsed routing did instead.
    """
    canto_host, aligned = _host_with_canto()
    host = build_default_fa_execution_host_v2(
        whisper_model=None,
        wave2vec_model=None,
        canto_host=canto_host,
        qwen_host=None,
    )
    assert host.canto_runner is not None

    payload = PreparedFaPayloadV2(
        words=["好", "唔該"],
        word_ids=["u0:w0", "u0:w1"],
        word_utterance_indices=[0, 0],
        word_utterance_word_indices=[0, 1],
    )
    request = ForcedAlignmentRequestV2.model_validate(
        {
            "backend": "wav2vec_canto",
            "payload_ref_id": "p1",
            "audio_ref_id": "a1",
            "text_mode": "char_joined",
            "pauses": False,
        }
    )

    timings = host.canto_runner(np.zeros(1600, dtype=np.float32), payload, request)

    assert aligned == [["jyut-好", "jyut-唔該"]], (
        "words reached the aligner unromanized"
    )
    assert len(timings) == 2
