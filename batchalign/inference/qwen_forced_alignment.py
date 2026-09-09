"""The Qwen3 forced aligner, shared by the Qwen3-ASR engine and the FA engine.

Two callers, one aligner:

- ``inference.languages.cantonese._qwen_common.QwenRecognizer`` loads it as the
  timestamp companion of the Qwen3-ASR model. Word-level timestamps are
  load-bearing there (the FA pipeline injects them into ``%wor``), which is why
  ``LoadedQwen`` has a single constructor that produces the ASR model AND the
  aligner or raises.
- ``worker._model_loading.forced_alignment`` loads it on its own, as the
  ``qwen3_fa`` forced-alignment engine, and aligns a transcript it is handed
  rather than one it produced.

Everything both callers need lives here so that a fix to the aligner call
reaches both. What deliberately does NOT live here is the conversion out of
window time, for the reason given on `WindowAlignedWord`.

Model weights are downloaded lazily by ``QwenForcedAligner.load``; nothing in
this module imports ``torch`` or ``transformers`` at import time, so the worker
bootstrap can reference it without paying for either.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, NamedTuple, NewType, Protocol

from batchalign.inference._domain_types import LanguageCode

L = logging.getLogger("batchalign.qwen.fa")


# Canonical forced-alignment companion from the Qwen3-ASR family, in the `-hf`
# spelling the native `transformers` module targets.
QWEN_FORCED_ALIGNER_MODEL_ID: Final[str] = "Qwen/Qwen3-ForcedAligner-0.6B-hf"


QwenLanguageLabel = NewType("QwenLanguageLabel", str)
"""A language label Qwen3 accepts, e.g. ``"Cantonese"``.

A `NewType` rather than a bare `str` because it is a closed set with exactly
one legitimate producer, `resolve_qwen_language`. Holding one is the proof that
the language was looked up rather than guessed: passing an ISO-639-3 code
straight through would be accepted by the model as an auto-detect request and
silently mis-classify short or low-energy audio.
"""


# ISO-639-3 -> the English language label Qwen3 expects. Pinned in code rather
# than via pycountry because pycountry returns "Yue Chinese" for `yue`, which
# Qwen does not accept (silent fall-through to auto-detect). The fix is an
# explicit per-code mapping with a fail-loud default.
#
# This is OUR supported set, and it is narrower than the aligner's own: the
# upstream checkpoint also advertises French, German, Italian, Japanese,
# Korean, Portuguese, Russian and Spanish. Widening this map is a measurement,
# not a typo fix, so it is left to whoever runs one.
QWEN_LANG_LABELS: Final[dict[LanguageCode, QwenLanguageLabel]] = {
    "yue": QwenLanguageLabel("Cantonese"),
    "zho": QwenLanguageLabel("Chinese"),
    "cmn": QwenLanguageLabel("Chinese"),
    "eng": QwenLanguageLabel("English"),
}


def resolve_qwen_language(lang: LanguageCode) -> QwenLanguageLabel:
    """The Qwen label for an ISO-639-3 code, or a named refusal.

    The only producer of a `QwenLanguageLabel`. The Rust control plane refuses
    an unsupported language at admission (`validate_fa_language_support`), so
    reaching this raise means the two lists have drifted apart.
    """
    label = QWEN_LANG_LABELS.get(lang)
    if label is None:
        raise ValueError(
            f"Qwen3 has no language label mapped for ISO-639-3 {lang!r}; "
            f"supported: {', '.join(sorted(QWEN_LANG_LABELS))}. Add it to "
            f"QWEN_LANG_LABELS in qwen_forced_alignment.py, and to "
            f"FA_QWEN3_LANGUAGES in crates/batchalign/src/types/engines.rs, "
            f"if the model supports the language."
        )
    return label


@dataclass(frozen=True, slots=True)
class WindowAlignedWord:
    """One aligned word, in the time of the AUDIO WINDOW it was aligned against.

    The type name carries the coordinate space on purpose. The aligner is
    handed a window (an ASR chunk, or an FA group's audio) and answers relative
    to the start of that window, never relative to the recording. Converting to
    recording time needs the window's own offset, which only the caller holds,
    so this type never converts and no consumer can mistake one space for the
    other.
    """

    text: str
    start_s: float
    end_s: float

    @classmethod
    def from_decoded(cls, unit: Mapping[str, Any]) -> WindowAlignedWord:
        """Read ONE decoded aligner unit. The boundary parse, and it is total.

        `decode_forced_alignment` documents `text`, `start_time` and
        `end_time`, and the shipped implementation writes all three. This read
        is nonetheless total over `text`, because it used to be: the ASR
        caller's own guard began `word.get("text", "") or ""`, so a unit with
        no text was skipped rather than crashing a chunk mid-transcription.
        Replacing that with `unit["text"]` moved a skip to a `KeyError` on a
        path that had worked for a year.

        The TIMES are different: they are refused by name rather than
        defaulted, because a unit carries its position in the FA fold and a
        dropped or zeroed one shifts every later span onto the wrong word.
        """
        for key in ("start_time", "end_time"):
            if unit.get(key) is None:
                raise QwenUnitNotTimed(key=key, unit=dict(unit))
        return cls(
            text=str(unit.get("text", "") or ""),
            start_s=float(unit["start_time"]),
            end_s=float(unit["end_time"]),
        )


@dataclass(frozen=True, slots=True)
class SpokenWindowWords:
    """The aligner units that carry TEXT, for a caller that reads them as words.

    A separate type from a `list[WindowAlignedWord]` because dropping a unit is
    exactly what the FA fold must never do and exactly what the ASR path must
    do. The fold attributes units to CHAT words BY POSITION, so a filtered list
    handed to it would move every later span onto the wrong word; the ASR path
    holds no positional contract at all and simply reports the words it was
    given timings for.

    Naming the filtered form is the guard the ASR path used to spell inline,
    restored as a transition instead of an `if` in a loop: `of_units` is the
    only way to get one, and nothing that takes a `WindowAlignment` takes one
    of these.
    """

    words: tuple[WindowAlignedWord, ...]

    @classmethod
    def of_units(cls, units: Iterable[WindowAlignedWord]) -> SpokenWindowWords:
        """Keep the units with text; a whitespace-only unit is not a word."""
        return cls(words=tuple(unit for unit in units if unit.text.strip()))


@dataclass(frozen=True, slots=True)
class WindowAlignment:
    """What the aligner segmented AND what it timed, from the SAME call.

    The two halves used to come from different calls: `WordSegmentation` asked
    `split_words`, and `align` let `prepare_forced_aligner_inputs` recompute
    the segmentation internally and threw its answer away. Nothing related the
    two beyond their length, so a segmentation that differed at the same COUNT
    folded silently onto the wrong words.

    Holding both together is the cure: possession of a `WindowAlignment` is the
    proof that these times belong to these units, and `WordSegmentation.fold`
    can check the units it proved against the units the times are actually for.
    """

    units: tuple[str, ...]
    """The aligner's own segmentation, as the aligning call used it."""
    words: tuple[WindowAlignedWord, ...]
    """One timed span per unit, in the same order."""

    @classmethod
    def of(
        cls, units: Iterable[str], words: Iterable[WindowAlignedWord]
    ) -> WindowAlignment:
        """The only constructor, and where the model-boundary arity is checked."""
        unit_tuple = tuple(str(unit) for unit in units)
        word_tuple = tuple(words)
        if len(unit_tuple) != len(word_tuple):
            raise QwenAlignmentMismatch(
                requested=list(unit_tuple),
                produced=[word.text for word in word_tuple],
            )
        return cls(units=unit_tuple, words=word_tuple)


class ForcedAligner(Protocol):
    """What both callers need from a loaded aligner.

    A protocol so a test can supply a recording double without loading 0.6B
    parameters, and so `LoadedQwen` can name the aligner in its own type.
    """

    def split_words(
        self, transcript: str, language: QwenLanguageLabel
    ) -> list[str]: ...

    def align(
        self, audio: Any, transcript: str, language: QwenLanguageLabel
    ) -> WindowAlignment: ...


class QwenUnitNotTimed(ValueError):
    """A decoded aligner unit arrived without one of its times.

    Named rather than left as the raw `KeyError` the dict access would raise,
    and refused rather than skipped: a unit holds a POSITION in the fold, so
    dropping it silently moves every later span onto the wrong CHAT word.
    """

    def __init__(self, key: str, unit: dict[str, Any]) -> None:
        super().__init__(
            f"Qwen3 decoded an alignment unit with no {key!r}: {unit!r}; the "
            f"unit cannot be dropped, because the fold attributes units to "
            f"CHAT words by position."
        )
        self.key = key
        self.unit = unit


class QwenSegmentationDrift(ValueError):
    """The aligner segmented one way when asked and another way when aligning.

    NOT an arity disagreement, which is what `QwenAlignmentMismatch` covers:
    the two segmentations can have the SAME length and different boundaries,
    and that case is invisible to a count. It matters because the fold's
    `unit_counts` come from the segmenting call while the spans come from the
    aligning call, so a drift attributes real times to the wrong words with
    nothing anywhere reporting it.
    """

    def __init__(self, proved: list[str], aligned: list[str]) -> None:
        super().__init__(
            f"Qwen3 segmented the transcript into {proved!r} when asked and "
            f"into {aligned!r} while aligning it; the spans belong to units "
            f"the fold was not proved against."
        )
        self.proved = proved
        self.aligned = aligned


class QwenSegmentationNotDecomposable(ValueError):
    """The aligner's tokenizer does not segment word by word.

    The fold below rests on ONE property of the aligner's tokenizer: segmenting
    the space-joined transcript gives the same units, in the same order, as
    segmenting each word on its own. It holds for the tokenizer every language
    we map uses (a space flushes its character buffer), and it does NOT hold
    for the Japanese and Korean branches, which run a morphological analyser
    over the whole string.

    So the property is CHECKED, per group, rather than assumed. Raising here
    means `QWEN_LANG_LABELS` has grown a language whose tokenizer is not
    word-decomposable, and the fold would silently mis-attribute spans.

    REACHABLE, and tested as such. `QWEN_LANG_LABELS` maps no whole-string
    analyser today, so the shipped `QwenForcedAligner` cannot trigger it; the
    `ForcedAligner` protocol admits any aligner, and a double that segments the
    joined string differently does trigger it
    (`test_a_whole_string_analyser_is_refused_before_any_forward_pass`). It is
    checked BEFORE the forward pass on purpose: the refusal costs a string
    split, not a second of GPU time.
    """

    def __init__(self, whole: list[str], per_word: list[str]) -> None:
        super().__init__(
            f"Qwen3 segmented the joined transcript into {whole!r} but its "
            f"words individually into {per_word!r}; this language's tokenizer "
            f"is not word-decomposable, so aligner units cannot be attributed "
            f"to CHAT words."
        )
        self.whole = whole
        self.per_word = per_word


class QwenAlignmentMismatch(ValueError):
    """The model timed a different number of units than it segmented.

    NOT a disagreement about words: that is what the fold below exists to
    absorb. This is the aligner answering with a unit count its own tokenizer
    did not produce, which would leave every unit after the divergence
    attributed to the wrong CHAT word. There is no correct silent handling, so
    it is refused with both lists in the message.

    A MODEL-BOUNDARY check, and it now lives on `WindowAlignment.of`, the one
    constructor that sees both the aligner's units and its timings. It is
    unreachable through the shipped `QwenForcedAligner`, whose
    `decode_forced_alignment` builds one entry per unit of `word_lists`; it is
    reachable through any other `ForcedAligner`, which is what the protocol
    admits, and through a future change to that upstream.

    Distinct from `QwenSegmentationDrift`, which is about the units themselves
    disagreeing between two calls at the same count. This one is about a
    disagreement a count CAN see.
    """

    def __init__(self, requested: list[str], produced: list[str]) -> None:
        super().__init__(
            f"Qwen3 forced alignment timed {len(produced)} unit(s) for "
            f"{len(requested)} unit(s) of its own segmentation; "
            f"requested={requested!r} produced={produced!r}"
        )
        self.requested = requested
        self.produced = produced


class UntimedReason(Enum):
    """Why one CHAT word came back without a timing.

    Exactly one variant, and that is a claim about the fold rather than an
    oversight: every word the aligner's tokenizer keeps at least one character
    of owns at least one unit, and every unit is timed, so the only way to
    reach the end of the fold with no span is to have contributed nothing to
    segment. A second variant would need a second way, and there is not one.
    """

    NO_ALIGNABLE_CHARACTERS = "no_alignable_characters"
    """The aligner's tokenizer kept none of this word's characters.

    Its rule keeps letters, digits, apostrophes and CJK; a CHAT word that is
    entirely something else (a bare marker, say) segments to nothing.
    """


@dataclass(frozen=True, slots=True)
class TimedChatWord:
    """One CHAT word with the span its aligner units cover, WINDOW-relative."""

    word: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True, slots=True)
class UntimedChatWord:
    """One CHAT word the aligner could not time, and why."""

    word: str
    reason: UntimedReason


FoldedChatWord = TimedChatWord | UntimedChatWord
"""A CHAT word's outcome. A sum type, so no caller can read a span that is not
there, and adding an outcome breaks every match rather than defaulting."""


@dataclass(frozen=True, slots=True)
class WordSegmentation:
    """Our CHAT words, and the aligner's OWN units, related word by word.

    Two spaces meet here: the CHAT words a caller handed over, and the units
    the aligner segments a transcript into. They are NOT the same space, which
    is the fact the engine used to get wrong: `你好` is one CHAT word and two
    aligner units, and `black+bird` is one CHAT word and one aligner unit
    spelled differently, because the aligner's tokenizer drops `+`.

    `of_words` is the sanctioned constructor, and it is where the invariant
    (that `units` and `unit_counts` were both segmented from `transcript`) is
    established, by checking the two segmentations against each other. Python
    cannot hide a dataclass's own constructor, so that is a check at ONE place
    rather than a proof no caller can forge; do not add a second way in.
    """

    words: tuple[str, ...]
    transcript: str
    units: tuple[str, ...]
    """Every aligner unit the transcript segments to, in order."""
    unit_counts: tuple[int, ...]
    """`unit_counts[i]` is how many of `units` CHAT word `i` contributed."""

    @classmethod
    def of_words(
        cls, words: list[str], aligner: ForcedAligner, language: QwenLanguageLabel
    ) -> WordSegmentation:
        """Segment each word with the ALIGNER'S OWN tokenizer, and prove it composes.

        Asking the aligner per word rather than reimplementing its rule is
        deliberate: its rule (`_is_kept_char` in
        `transformers/models/qwen3_asr/processing_qwen3_asr.py`) is theirs to
        change, and a copy of it here would be a second thing to keep true.

        Pure string work, so this costs a split rather than a forward pass.

        COST, AND WHY IT STAYS. This makes W+1 tokenizer calls per group (one
        per word, plus one for the whole transcript), which an efficiency
        review flagged as avoidable. It stays, deliberately, because both
        cheaper shapes buy CPU with a weaker proof:

        - Validating against the `word_lists` the forward pass itself returns
          makes `fold`'s drift check compare a value with itself, so a
          same-count different-boundary segmentation would fold onto the wrong
          words silently, and the refusal would come after the forward pass
          rather than before it.
        - Deriving the per-word counts greedily from `whole` alone weakens the
          claim from "the per-word segmentations CONCATENATE to the whole one"
          to "the whole one PARTITIONS into per-word character runs", which
          admits a context-sensitive tokenizer that this check refuses today.

        W+1 string splits are small beside the GPU forward pass they precede,
        and the proof is what stops a wrong-word timing. Measured and decided
        2026-09-07; do not re-propose without a measurement showing the splits
        actually dominate.
        """
        transcript = " ".join(words).strip()
        per_word = [aligner.split_words(word, language) for word in words]
        whole = aligner.split_words(transcript, language)
        flattened = [unit for units in per_word for unit in units]
        if whole != flattened:
            raise QwenSegmentationNotDecomposable(whole=whole, per_word=flattened)
        return cls(
            words=tuple(words),
            transcript=transcript,
            units=tuple(whole),
            unit_counts=tuple(len(units) for units in per_word),
        )

    def fold(self, alignment: WindowAlignment) -> QwenWordFold:
        """Fold aligner units back onto CHAT words: the named transition.

        A word's span is its FIRST constituent unit's start and its LAST
        constituent unit's end, which is containment and nothing cleverer: the
        units of a word are contiguous because the transcript is space-joined
        and a space ends a unit.

        A word with no constituent unit is `UntimedChatWord`. It is never
        padded from a neighbour: downstream, a fabricated timing looks exactly
        like a measured one.

        The units the alignment carries are the ones its times are FOR, so the
        check is an equality against the units this segmentation was proved on,
        not a comparison of two lengths. A count cannot see a same-length
        boundary shift, and a boundary shift is what silently moves spans.
        """
        aligned = list(alignment.words)
        if list(alignment.units) != list(self.units):
            raise QwenSegmentationDrift(
                proved=list(self.units),
                aligned=list(alignment.units),
            )

        folded: list[FoldedChatWord] = []
        cursor = 0
        for word, count in zip(self.words, self.unit_counts, strict=True):
            if count == 0:
                folded.append(
                    UntimedChatWord(
                        word=word, reason=UntimedReason.NO_ALIGNABLE_CHARACTERS
                    )
                )
                continue
            constituents = aligned[cursor : cursor + count]
            cursor += count
            folded.append(
                TimedChatWord(
                    word=word,
                    start_ms=int(round(constituents[0].start_s * 1000)),
                    end_ms=int(round(constituents[-1].end_s * 1000)),
                )
            )
        return QwenWordFold(words=tuple(folded))


class IndexedWordTimingRow(NamedTuple):
    """One row of the worker protocol's indexed-timing host output.

    The shape `crates/batchalign-pyo3/src/worker_fa_exec.rs::parse_indexed_timings`
    reads: word text, the interval, and an optional model score. `interval_ms`
    is `None` for a word the aligner could not time, which the Rust side turns
    into `indexed_timings[i] = None`, the wire's existing and honest spelling
    of "this word has no timing".

    The wire has no slot for the REASON, and that is a deliberate stop rather
    than an oversight: `IndexedWordTimingResultV2.indexed_timings` is a
    `Vec<Option<IndexedWordTimingV2>>` all four FA backends share, and the only
    consumer downstream of it (`apply_indexed_timings`) leaves an untimed word
    without a bullet whatever the reason was. The reason is therefore typed and
    inspectable HERE, on `QwenWordFold`, where a test or a caller in this
    process can act on it, and the read-back on the far side is UNKNOWN rather
    than a fabricated span.
    """

    word: str
    interval_ms: tuple[int, int] | None
    model_score: float | None


@dataclass(frozen=True, slots=True)
class QwenWordFold:
    """One outcome per requested CHAT word, in the order they were requested.

    The RETURN of `QwenFaHost.align_words`, and the point of it: a word the
    aligner could not reach comes back as an `UntimedChatWord` rather than
    taking the whole group down with an exception, so a caller sees which
    words are unaligned and why. Before the fold, one unreachable word made
    the group's every word untimed and sent the reason to a `warn!`.
    """

    words: tuple[FoldedChatWord, ...]

    def to_indexed_timings(self) -> list[IndexedWordTimingRow]:
        """Lower the fold to the worker protocol's rows. The ONLY lossy step.

        Named, and called exactly once (`batchalign/worker/_fa_v2.py`), so the
        place where the reason stops travelling is a signature rather than a
        habit.
        """
        return [
            IndexedWordTimingRow(
                word=word.word,
                interval_ms=(
                    (word.start_ms, word.end_ms)
                    if isinstance(word, TimedChatWord)
                    else None
                ),
                # The aligner reports no per-word score, and a fabricated one
                # would be indistinguishable from a measured one downstream.
                model_score=None,
            )
            for word in self.words
        ]


@dataclass(frozen=True, slots=True)
class QwenForcedAligner:
    """The loaded Qwen3 forced-alignment model and its processor.

    Both fields are `Any` because `transformers` is imported lazily; the
    protocol above is what callers program against.
    """

    processor: Any
    model: Any

    @classmethod
    def load(cls, aligner_id: str, device: str, dtype: Any) -> QwenForcedAligner:
        """Load the aligner, or raise. There is no half-loaded aligner."""
        from transformers import (  # type: ignore[import-not-found]
            AutoProcessor,
            Qwen3ASRForTokenClassification,
        )

        return cls(
            processor=AutoProcessor.from_pretrained(aligner_id),
            model=Qwen3ASRForTokenClassification.from_pretrained(
                aligner_id, device_map=device, dtype=dtype
            ),
        )

    def split_words(self, transcript: str, language: QwenLanguageLabel) -> list[str]:
        """The aligner's OWN tokenization of a transcript.

        Exposed so a caller can check the segmentation BEFORE spending a
        forward pass on it. `transformers` computes exactly this inside
        `prepare_forced_aligner_inputs`; asking for it here is a pure function
        of the string.
        """
        words: list[str] = self.processor.split_words_for_alignment(
            transcript, str(language)
        )
        return words

    def align(
        self, audio: Any, transcript: str, language: QwenLanguageLabel
    ) -> WindowAlignment:
        """Align one transcript against one audio WINDOW.

        Times are window-relative; see `WindowAlignedWord`. One unit out per
        unit of the segmentation, in order, with nothing dropped: the caller
        folds these onto its own words by position, so a silent drop here would
        shift every span after it onto the wrong word. That is why the ASR
        path's blank-unit filter is NOT here but in `SpokenWindowWords`, which
        only the caller that ignores position uses.

        The returned `WindowAlignment` carries the segmentation
        `prepare_forced_aligner_inputs` actually used. Returning it rather than
        discarding it is the point: the fold's unit counts came from a separate
        `split_words` call, and until this returned `word_lists` nothing could
        tell that the two calls had agreed on anything but a length.
        """
        import torch  # type: ignore[import-not-found]

        inputs, word_lists = self.processor.prepare_forced_aligner_inputs(
            audio=audio, transcript=transcript, language=str(language)
        )
        inputs = inputs.to(self.model.device, self.model.dtype)
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        aligned = self.processor.decode_forced_alignment(
            logits=logits,
            input_ids=inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=self.model.config.timestamp_token_id,
        )[0]

        # One transcript in, so one word list out; `prepare_forced_aligner_inputs`
        # refuses a transcript/audio count mismatch itself.
        return WindowAlignment.of(
            units=word_lists[0],
            words=[WindowAlignedWord.from_decoded(unit) for unit in aligned],
        )


@dataclass(frozen=True, slots=True)
class QwenFaHost:
    """The standalone `qwen3_fa` forced-alignment engine's runtime bundle.

    Mirrors `CantoneseFaHost`: the language is fixed at load time from the
    worker's bootstrap language, so it never travels on the FA wire, and the
    host owns the whole verb (`align_words`) rather than exposing the pieces
    for each caller to reassemble.
    """

    aligner: ForcedAligner
    language: QwenLanguageLabel

    def segment(self, words: list[str]) -> WordSegmentation:
        """The aligner's OWN segmentation of these words, with no forward pass.

        Exposed because the fold's two counts (our words, its units) are what
        a reviewer has to see to believe the engine works on real input;
        `scripts/prove_qwen3_fa.py` reports them. Pure string work, so asking
        twice costs a split.
        """
        return WordSegmentation.of_words(words, self.aligner, self.language)

    def align_words(self, audio: Any, words: list[str]) -> QwenWordFold:
        """One outcome per requested word, timed in WINDOW milliseconds.

        Window-to-file conversion is the Rust control plane's job for FA
        groups (`crates/batchalign/src/chat_ops/fa/coordinates.rs`), so this
        deliberately does not do it. Doing it here as well would double-count
        the group offset.

        Three steps, and the middle one is the expensive one: segment (pure
        string work, so a language whose tokenizer cannot be folded is refused
        before a forward pass), align, fold. The aligner's units and our CHAT
        words are different spaces, so the fold is the only route between them.
        """
        segmentation = self.segment(words)
        alignment = self.aligner.align(audio, segmentation.transcript, self.language)
        return segmentation.fold(alignment)


def load_qwen_fa(lang: LanguageCode, *, device_policy: Any = None) -> QwenFaHost:
    """Load the standalone Qwen3 FA engine for one worker.

    The language is resolved FIRST, so an unsupported one fails at worker
    bootstrap rather than after a multi-gigabyte download.
    """
    import torch  # type: ignore[import-not-found]

    from batchalign.device import resolve_inference_device
    from batchalign.worker._progress import emit_hf_download_if_missing

    language = resolve_qwen_language(lang)
    device = resolve_inference_device(device_policy)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    emit_hf_download_if_missing(QWEN_FORCED_ALIGNER_MODEL_ID, kind="forced alignment")
    aligner = QwenForcedAligner.load(
        aligner_id=QWEN_FORCED_ALIGNER_MODEL_ID,
        device=device.type,
        dtype=dtype,
    )
    L.info(
        "Qwen3 FA loaded: aligner=%s, lang=%s (%s), device=%s, dtype=%s",
        QWEN_FORCED_ALIGNER_MODEL_ID,
        lang,
        language,
        device.type,
        dtype,
    )
    return QwenFaHost(aligner=aligner, language=language)


__all__ = [
    "QWEN_FORCED_ALIGNER_MODEL_ID",
    "QWEN_LANG_LABELS",
    "FoldedChatWord",
    "ForcedAligner",
    "IndexedWordTimingRow",
    "QwenAlignmentMismatch",
    "QwenFaHost",
    "QwenForcedAligner",
    "QwenLanguageLabel",
    "QwenSegmentationDrift",
    "QwenSegmentationNotDecomposable",
    "QwenUnitNotTimed",
    "QwenWordFold",
    "SpokenWindowWords",
    "TimedChatWord",
    "UntimedChatWord",
    "UntimedReason",
    "WindowAlignedWord",
    "WindowAlignment",
    "WordSegmentation",
    "load_qwen_fa",
    "resolve_qwen_language",
]
