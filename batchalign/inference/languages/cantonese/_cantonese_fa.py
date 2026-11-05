"""Cantonese FA inference provider for built-in HK/Cantonese engines.

The FA load builds a :class:`CantoneseFaHost` (``load_cantonese_fa``) and the
worker's load registry records it; a request aligns through that host's
``align_words``. Nothing is held at module level.

Wraps the standard Wave2Vec MMS forced alignment model with a Cantonese
preprocessing step: hanzi characters are converted to jyutping romanization
(tone-stripped, syllables joined with apostrophes) before alignment.  For
non-``yue`` languages, words pass through unchanged.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, assert_never

from batchalign.inference._admission import _FactoryAdmission
from batchalign.inference._domain_types import LanguageCode
from batchalign.worker._model_errors import ModelOutputInvalidError

from ._common import EngineOverrides

if TYPE_CHECKING:
    from batchalign.inference.types import Wave2VecFAHandle, Wave2VecFAResult

L = logging.getLogger("batchalign.hk.fa")

CANTONESE_FA_ENGINE_ID = "wav2vec-canto-v2-source-complete"
"""Romanization preserves every source span; old v1 results are not reused."""


# ---------------------------------------------------------------------------
# Protocol for pycantonese (avoids importing the heavy module at type level)
# ---------------------------------------------------------------------------


class _PyCantonese(Protocol):
    def characters_to_jyutping(self, text: str) -> list[tuple[str, str | None]]: ...


class _Wave2VecFaRunner(Protocol):
    """Callable signature for running Wave2Vec FA over one audio chunk."""

    def __call__(
        self,
        model: object,
        audio: object,
        words: list[str],
    ) -> Wave2VecFAResult: ...


@dataclass(frozen=True, slots=True)
class CantoneseFaHost:
    """What the Cantonese FA load built: the aligner and the romanizer.

    ``load_cantonese_fa`` builds it; tests build one around fakes. ``lang`` is
    the worker's language, fixed at load: words are romanized only for
    Cantonese.
    """

    model: Wave2VecFAHandle
    romanizer: _PyCantonese
    infer_wave2vec_fa: _Wave2VecFaRunner
    lang: LanguageCode

    def romanize_words(self, words: list[str]) -> list[str]:
        """The single shaping verb both preparation and execution consume."""
        return [_maybe_romanize(self.romanizer, word, self.lang) for word in words]

    def align_words(self, audio: object, words: list[str]) -> Wave2VecFAResult:
        """Romanize, then align, on audio the caller has already decoded.

        Owning the verb here stops each caller reassembling it out of this
        host's fields, which is how two callers once came to romanize by
        different rules.
        """
        romanized = self.romanize_words(words)
        return self.infer_wave2vec_fa(self.model, audio, romanized)


# ---------------------------------------------------------------------------
# Jyutping conversion
# ---------------------------------------------------------------------------


class RomanizationCoverageError(ModelOutputInvalidError):
    """The romanizer's result cannot account for the requested source word."""


@dataclass(frozen=True, slots=True)
class _PronouncedSpan:
    source: str
    syllables: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _UnmappedSpan:
    source: str


type _RomanizationSpan = _PronouncedSpan | _UnmappedSpan


@dataclass(frozen=True, slots=True, init=False)
class _CoveredRomanization(_FactoryAdmission):
    """An ordered partition whose source concatenates to the complete word."""

    spans: tuple[_RomanizationSpan, ...]

    @classmethod
    def admit(cls, text: str, observations: object) -> _CoveredRomanization:
        if not isinstance(observations, list):
            raise RomanizationCoverageError("romanizer source spans are not a list")
        spans: list[_RomanizationSpan] = []
        for observation in observations:
            if not isinstance(observation, tuple) or len(observation) != 2:
                raise RomanizationCoverageError(
                    "romanizer returned a malformed source span"
                )
            source, pronunciation = observation
            if not isinstance(source, str) or not source:
                raise RomanizationCoverageError(
                    "romanizer returned an empty or malformed source"
                )
            match pronunciation:
                case None:
                    spans.append(_UnmappedSpan(source))
                case str():
                    # Only a pronunciation has tone digits. Unmapped source
                    # text is retained verbatim, never normalized as Jyutping.
                    syllables = tuple(
                        re.sub(r"[0-9]", "", part) for part in pronunciation.split()
                    )
                    if not syllables or any(not part for part in syllables):
                        raise RomanizationCoverageError(
                            "romanizer returned an empty pronunciation"
                        )
                    spans.append(_PronouncedSpan(source, syllables))
                case _:
                    raise RomanizationCoverageError(
                        "romanizer returned a malformed pronunciation"
                    )
        if "".join(span.source for span in spans) != text:
            raise RomanizationCoverageError(
                "romanizer source spans do not cover the requested word"
            )
        value = object.__new__(cls)
        object.__setattr__(value, "spans", tuple(spans))
        return value

    def render(self) -> str:
        labels: list[str] = []
        for span in self.spans:
            match span:
                case _PronouncedSpan(syllables=syllables):
                    labels.extend(syllables)
                case _UnmappedSpan(source=source):
                    labels.append(source)
                case _:
                    assert_never(span)
        return "'".join(labels)


def _hanzi_to_jyutping(pc: _PyCantonese, text: str) -> str:
    """Convert Cantonese hanzi token to MMS-alignment jyutping form.

    - Uses ``pycantonese.characters_to_jyutping()`` to obtain romanization.
    - Strips tone digits (0-9).
    - Separates every SYLLABLE with an apostrophe (``"gwong'dung'waa"``).
    - Unknown characters (where pycantonese returns ``None``) pass through
      unchanged.

    Every returned source span must reconstruct the complete requested word.
    An explicitly unmapped span passes through; malformed or missing source
    data is a model-output refusal rather than a guessed successful conversion.

    THE SYLLABLE GUARANTEE REQUIRES pycantonese >= 5, which is why the floor in
    ``pyproject.toml`` is not decorative. Under 4.x this function claimed the
    same guarantee and could not deliver it: ``characters_to_jyutping`` ran the
    syllables of a single word together (``'gwong2dung1waa2'``) and separated
    only what its own segmenter split, so an apostrophe marked a WORD boundary
    that pycantonese happened to find, not a syllable. The result was
    inconsistent in a way no caller could see or predict: ``你好`` segmented
    into two words and romanized as ``nei'hou``, while ``廣東話`` stayed one
    word and became ``gwongdungwaa``. Both are single CHAT words here, since
    this is called once per word.

    5.0.0 space-separates syllables within a word, so the whitespace collapse
    below now marks exactly what this docstring always said it did.
    """
    if not text:
        return text
    return _CoveredRomanization.admit(text, pc.characters_to_jyutping(text)).render()


def _maybe_romanize(pc: _PyCantonese, word: str, lang: LanguageCode) -> str:
    """Romanize *word* only when *lang* is Cantonese (``"yue"``)."""
    if lang == "yue":
        return _hanzi_to_jyutping(pc, word)
    return word


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_cantonese_fa(
    lang: LanguageCode,
    engine_overrides: EngineOverrides | None,
    *,
    device_policy=None,
) -> CantoneseFaHost:
    """Load the Wave2Vec FA model and pycantonese converter for one FA load.

    Parameters
    ----------
    lang : str
        ISO 639-3 language code (e.g. ``"yue"``), fixed on the host.
    engine_overrides : EngineOverrides or None
        Engine overrides (currently unused, reserved for future
        model selection).
    """
    try:
        import pycantonese as pc
    except ImportError as exc:
        raise ImportError(
            "Cantonese FA dependency 'pycantonese' is missing from this "
            "environment. Reinstall batchalign3 or install pycantonese."
        ) from exc

    from batchalign.inference.fa import infer_wave2vec_fa, load_wave2vec_fa

    model = load_wave2vec_fa(device_policy=device_policy)
    L.info("Cantonese FA model loaded: lang=%s", lang)
    return CantoneseFaHost(
        model=model,
        romanizer=pc,
        infer_wave2vec_fa=infer_wave2vec_fa,
        lang=lang,
    )


__all__ = [
    "CantoneseFaHost",
    "_hanzi_to_jyutping",
    "_maybe_romanize",
    "load_cantonese_fa",
]
