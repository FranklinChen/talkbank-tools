"""Forced-alignment confirmation verify engine: does the placed text align?

One of the three placement-verification engines behind `merge-verify`
(with pitch banding and the machine ear). Aligns a placed utterance's
words against the audio window around its bullet span with the
torchaudio MMS_FA bundle and reports the mean/min per-word alignment
score.

CALIBRATION-LOCKED: every constant here (sample rate, window padding
and bounds, the star-token model variant, CPU inference, the word
normalization alphabet) matches the pipeline that was human-calibrated
in July 2026. Changing any of them, or substituting a different
aligner, invalidates that calibration; recalibrate against blind
listening verdicts before shipping such a change.

THE SCORE IS ORDERING-ONLY, NEVER A GATE. Calibration measured that
low FA score does not mean wrong placement (11/18 sampled FA-fail
lines were correctly placed: disfluent child speech aligns poorly) and
high FA score does not mean right placement (0.94 measured on pure
adult speech: the aligner happily aligns the wrong voice). The score
orders the human review queue worst-first; the promote/demote decision
belongs to the composed rule (pitch band + machine ear).
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    import torch

SAMPLE_RATE = 16_000
# Window shaping around the placed bullet span: pad so word onsets
# survive placement jitter, floor so short utterances get enough
# acoustic context to score meaningfully, cap so a corrupt span cannot
# demand minutes of audio.
WINDOW_PAD_MS = 1200
WINDOW_MIN_MS = 1500
WINDOW_MAX_MS = 30_000
# The calibrated 5th-percentile-of-confident-class threshold. Kept for
# reference and for ordering displays; per the module docstring it is
# NEVER a promote/demote gate.
FA_ORDERING_THRESHOLD = 0.114


class ScoredTokenSpan(Protocol):
    """One aligned token span as returned by the MMS_FA aligner."""

    @property
    def score(self) -> float: ...


class FaTokenizer(Protocol):
    """The MMS_FA bundle tokenizer: words in, aligner-ready tokens out."""

    def __call__(self, transcript: list[str]) -> object: ...


class FaAligner(Protocol):
    """The MMS_FA bundle aligner: emission + tokens in, per-word spans out."""

    def __call__(
        self, emission: torch.Tensor, tokens: object
    ) -> Sequence[Sequence[ScoredTokenSpan]]: ...


class FaEmissionModel(Protocol):
    """The acoustic model: batched waveform in, (emission, lengths) out."""

    def __call__(self, waveform: torch.Tensor) -> tuple[torch.Tensor, object]: ...


class FaConfirmHandle:
    """Loaded MMS_FA bundle pieces needed to score windows.

    Named fields instead of a (model, tokenizer, aligner) tuple; see the
    cross-repo no-tuple-seams rule.
    """

    def __init__(
        self,
        model: FaEmissionModel,
        tokenizer: FaTokenizer,
        aligner: FaAligner,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.aligner = aligner


@dataclass(frozen=True, slots=True)
class WindowFaScore:
    """Alignment quality of one placed window (ordering-only, see module doc)."""

    word_count: int
    mean_word_score: float
    min_word_score: float


def normalize_word(word: str) -> str:
    """Lowercase a word to the aligner's a-z' alphabet; empty if nothing survives."""
    return "".join(
        ch for ch in word.lower() if ch.isascii() and (ch.isalpha() or ch == "'")
    )


def normalize_words(words: Sequence[str]) -> tuple[str, ...]:
    """Normalize a token sequence for alignment, dropping words that vanish."""
    return tuple(w for w in (normalize_word(t) for t in words) if w)


def confirm_window_span(start_ms: int, end_ms: int) -> tuple[int, int]:
    """Padded, length-bounded audio window (ms) for one placed bullet span."""
    start = max(0, start_ms - WINDOW_PAD_MS)
    end = end_ms + WINDOW_PAD_MS
    if end - start < WINDOW_MIN_MS:
        deficit = WINDOW_MIN_MS - (end - start)
        start = max(0, start - deficit // 2)
        end = start + WINDOW_MIN_MS
    if end - start > WINDOW_MAX_MS:
        end = start + WINDOW_MAX_MS
    return start, end


def load_fa_confirm() -> FaConfirmHandle:
    """Load the MMS_FA bundle for confirmation scoring (CPU, star token).

    ``with_star=True`` and CPU inference are calibration-locked: the
    star token absorbs unspoken material inside the padded window, and
    the calibration scores were produced on CPU.
    """
    # Heavy imports kept local so module import stays cheap for callers
    # that only need the window math and score types.
    import torch
    from torchaudio.pipelines import MMS_FA as bundle

    from batchalign.inference.fa import mms_fa_model

    model = mms_fa_model(purpose="placement confirmation").to(torch.device("cpu"))
    return FaConfirmHandle(
        model=model,
        tokenizer=bundle.get_tokenizer(),
        aligner=bundle.get_aligner(),
    )


def score_window(
    handle: FaConfirmHandle,
    audio: np.ndarray,
    sample_rate: int,
    words: Sequence[str],
) -> WindowFaScore:
    """MMS_FA-align normalized words against one window of mono audio.

    ``audio`` is the float mono window itself (callers cut the
    `confirm_window_span` region upstream) at `sample_rate`; a
    mismatched rate raises rather than silently mis-scoring, because
    the acoustic model is calibration-locked to 16 kHz. ``words`` must
    be non-empty and already normalized (`normalize_words`); an empty
    sequence raises because a scoreless line must be routed by its
    category, never given a fabricated score.
    """
    import torch

    if sample_rate != SAMPLE_RATE:
        msg = f"FA confirmation is calibrated at {SAMPLE_RATE} Hz; got {sample_rate}"
        raise ValueError(msg)
    if not words:
        msg = "FA confirmation needs at least one alignable word"
        raise ValueError(msg)

    waveform = torch.from_numpy(audio).unsqueeze(0)
    with torch.inference_mode():
        emission, _ = handle.model(waveform)
        token_spans = handle.aligner(emission[0], handle.tokenizer(list(words)))
    word_scores = [
        sum(s.score for s in spans) / max(1, len(spans)) for spans in token_spans
    ]
    return WindowFaScore(
        word_count=len(word_scores),
        mean_word_score=statistics.fmean(word_scores),
        min_word_score=min(word_scores),
    )
