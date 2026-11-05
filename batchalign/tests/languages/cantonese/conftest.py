"""Shared fixtures and test doubles for Cantonese engine tests."""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from batchalign.inference.languages.cantonese._cantonese_fa import CantoneseFaHost

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
CLIP_MP3 = FIXTURES / "05b_clip.mp3"
CLIP_WAV = FIXTURES / "05b_clip.wav"


# ---------------------------------------------------------------------------
# PyCantonese test double
# ---------------------------------------------------------------------------

# Deterministic jyutping lookup matching pycantonese.characters_to_jyutping()
_JYUTPING: dict[str, str] = {
    "你": "nei5",
    "好": "hou2",
    "我": "ngo5",
    "係": "hai6",
    "咁": "gam3",
    "搞": "gaau2",
    "笑": "siu3",
}


class PyCantoneseFake:
    """Test double implementing the pycantonese Protocol used by _cantonese_fa."""

    @staticmethod
    def characters_to_jyutping(text: str) -> list[tuple[str, str | None]]:
        return [(c, _JYUTPING.get(c)) for c in text]


@pytest.fixture
def pc_fake() -> PyCantoneseFake:
    return PyCantoneseFake()


@pytest.fixture
def pc_real():
    """Provide the real pycantonese module for jyutping tests.

    PyCantonese is a fast pure-Python library, no reason to fake it.
    Tests using this fixture exercise the actual jyutping dictionary.
    """
    import pycantonese

    return pycantonese


# ---------------------------------------------------------------------------
# Cantonese FA model fakes
# ---------------------------------------------------------------------------


def _fake_infer_wave2vec_fa(
    model: Any, audio: Any, words: list[str]
) -> list[tuple[str, tuple[int, int], float | None]]:
    """Deterministic FA: word i gets timing (i*100, (i+1)*100)."""
    return [(w, (i * 100, (i + 1) * 100), None) for i, w in enumerate(words)]


@pytest.fixture
def cantonese_fa_host() -> CantoneseFaHost:
    """Provide an explicit fake Cantonese FA host for unit tests."""
    return CantoneseFaHost(
        model=object(),
        romanizer=PyCantoneseFake(),
        infer_wave2vec_fa=_fake_infer_wave2vec_fa,
        lang="yue",
    )
