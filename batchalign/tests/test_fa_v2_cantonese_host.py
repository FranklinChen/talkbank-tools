# affects: batchalign/worker/_fa_v2.py
"""The Cantonese FA load's host must be servable by the V2 FA executor.

The executor once had a matching dispatch arm while nothing built the runner,
so a Cantonese request could not be served even once the routing constructed
one. The jyutping conversion was unit-tested throughout; what was missing was
the assembly. The FA load now builds the runner over the host it loaded
(``cantonese_fa_runner``), with the word-list call shape every word aligner
shares.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from batchalign.worker._fa_v2 import cantonese_fa_runner


class _FakeRomanizer:
    """Stands in for pycantonese, recording what it was asked to convert."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def characters_to_jyutping(self, word: str) -> list[tuple[str, str]]:
        self.seen.append(word)
        return [(word, f"jyut-{word}")]


def _host_with_canto(lang: str = "yue") -> tuple[Any, list[list[str]]]:
    """A Cantonese host whose aligner records the words it actually received."""
    from batchalign.inference.languages.cantonese._cantonese_fa import CantoneseFaHost

    aligned: list[list[str]] = []

    def _fake_align(_model: object, _audio: object, words: list[str]):
        aligned.append(list(words))
        return [(w, (i * 100, i * 100 + 90), None) for i, w in enumerate(words)]

    host = CantoneseFaHost(
        model=object(),
        romanizer=_FakeRomanizer(),
        infer_wave2vec_fa=_fake_align,
        lang=lang,
    )
    return host, aligned


def test_the_runner_romanizes_before_aligning() -> None:
    """Jyutping conversion is the whole reason this engine exists.

    Aligning Han characters directly against a wav2vec model trained on
    romanized Cantonese is what selecting `wav2vec_canto` is meant to avoid,
    and is what the collapsed routing did instead.
    """
    canto_host, aligned = _host_with_canto()
    runner = cantonese_fa_runner(canto_host)

    timings = runner(np.zeros(1600, dtype=np.float32), ["好", "唔該"])

    assert aligned == [["jyut-好", "jyut-唔該"]], (
        "words reached the aligner unromanized"
    )
    assert len(timings) == 2


def test_a_non_cantonese_host_aligns_the_words_unchanged() -> None:
    """The host's language is the one its load was given: words are romanized
    only for Cantonese, as the host documents."""
    canto_host, aligned = _host_with_canto(lang="eng")

    cantonese_fa_runner(canto_host)(np.zeros(1600, dtype=np.float32), ["hello"])

    assert aligned == [["hello"]]
