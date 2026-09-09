"""The Qwen3 forced aligner is ONE module with two callers.

Origin: the Qwen3 aligner arrived as an internal companion of the Cantonese
Qwen3-ASR engine, reachable only by transcribing. Exposing it as a standalone
FA engine could have been done by copying the twelve lines that call it, which
is how two copies of a coordinate conversion come to disagree. It is a shared
module instead, and these tests pin the two properties that matter about the
sharing:

1. Both callers go through the SAME ``align`` seam, so a fix to the aligner
   call reaches both.
2. The window-to-file conversion happens in exactly ONE of them. The aligner
   answers in WINDOW time always; the ASR chunk path converts to recording
   time because it owns the chunk offset, and the FA path does NOT, because
   the Rust control plane owns that conversion for FA groups
   (``crates/batchalign/src/chat_ops/fa/coordinates.rs``). A second conversion
   here would double-count the offset.

These are behaviours a signature cannot describe (which of two callers applies
a conversion), so they stay tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from batchalign.inference.languages.cantonese import _qwen_common as qwen_common
from batchalign.inference.languages.cantonese._qwen_chunking import (
    SAMPLE_RATE,
    AudioChunk,
)
from batchalign.inference.qwen_forced_alignment import (
    QwenAlignmentMismatch,
    QwenFaHost,
    QwenLanguageLabel,
    QwenSegmentationDrift,
    QwenSegmentationNotDecomposable,
    QwenUnitNotTimed,
    UntimedReason,
    WindowAlignedWord,
    WindowAlignment,
    resolve_qwen_language,
)


class _RecordingAligner:
    """A stand-in for the loaded Qwen aligner that records what it was asked.

    Tokenizes on whitespace, which is what the real aligner's
    ``split_words_for_alignment`` does for space-delimited scripts.
    """

    def __init__(self, aligned: list[WindowAlignedWord]) -> None:
        self.aligned = aligned
        self.calls: list[tuple[str, str]] = []

    def split_words(self, transcript: str, language: QwenLanguageLabel) -> list[str]:
        return transcript.split()

    def align(
        self, audio: object, transcript: str, language: QwenLanguageLabel
    ) -> WindowAlignment:
        self.calls.append((transcript, str(language)))
        return WindowAlignment.of(
            units=[word.text for word in self.aligned], words=self.aligned
        )


def _loaded_qwen(aligner: _RecordingAligner) -> qwen_common.LoadedQwen:
    """A ``LoadedQwen`` whose ASR half is inert and whose aligner is the fake."""
    return qwen_common.LoadedQwen(processor=object(), model=object(), aligner=aligner)


def test_both_callers_reach_the_aligner_through_the_same_seam() -> None:
    aligner = _RecordingAligner(
        [
            WindowAlignedWord(text="hello", start_s=0.10, end_s=0.40),
            WindowAlignedWord(text="world", start_s=0.50, end_s=0.90),
        ]
    )

    recognizer = qwen_common.QwenRecognizer(
        lang="eng", model_id="Qwen/Qwen3-ASR-0.6B-hf", device="cpu"
    )
    chunk = AudioChunk(samples=np.zeros(SAMPLE_RATE, dtype=np.float32), offset=0.0)
    recognizer._align_chunk(_loaded_qwen(aligner), chunk, "hello world")

    host = QwenFaHost(aligner=aligner, language=resolve_qwen_language("eng"))
    host.align_words(np.zeros(SAMPLE_RATE, dtype=np.float32), ["hello", "world"])

    assert [transcript for transcript, _ in aligner.calls] == [
        "hello world",
        "hello world",
    ]


def test_only_the_asr_chunk_path_converts_window_time_to_file_time() -> None:
    aligned = [WindowAlignedWord(text="hello", start_s=0.10, end_s=0.40)]

    recognizer = qwen_common.QwenRecognizer(
        lang="eng", model_id="Qwen/Qwen3-ASR-0.6B-hf", device="cpu"
    )
    chunk = AudioChunk(samples=np.zeros(SAMPLE_RATE, dtype=np.float32), offset=12.0)
    asr_timings = recognizer._align_chunk(
        _loaded_qwen(_RecordingAligner(aligned)), chunk, "hello"
    )
    assert asr_timings == [(12.10, 12.40, "hello")]

    host = QwenFaHost(
        aligner=_RecordingAligner(aligned), language=resolve_qwen_language("eng")
    )
    fa_fold = host.align_words(np.zeros(SAMPLE_RATE, dtype=np.float32), ["hello"])
    assert [row.interval_ms for row in fa_fold.to_indexed_timings()] == [(100, 400)]


def test_a_language_outside_the_label_map_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="deu"):
        resolve_qwen_language("deu")


def test_a_unit_count_the_model_did_not_segment_is_refused_not_padded() -> None:
    """A DIFFERENT refusal from the one the fold absorbed.

    A word list the aligner segments differently is now folded, not refused.
    What is still refused is the model timing a number of units its own
    tokenizer did not produce: the fold attributes units to words by position,
    so a miscount would silently shift every span after it onto the wrong
    word."""
    aligner = _RecordingAligner(
        [WindowAlignedWord(text="helloworld", start_s=0.0, end_s=1.0)]
    )
    host = QwenFaHost(aligner=aligner, language=resolve_qwen_language("eng"))

    with pytest.raises(QwenSegmentationDrift):
        host.align_words(
            np.zeros(SAMPLE_RATE, dtype=np.float32), ["hello", "world", "again"]
        )


def test_a_model_that_times_more_units_than_it_segmented_is_refused() -> None:
    """The MODEL-boundary arity check, at the one constructor that sees both.

    Distinct from the drift above: this is the aligner disagreeing with
    ITSELF, timing a unit count its own segmentation did not produce, which is
    what `QwenAlignmentMismatch` has always been about.
    """
    with pytest.raises(QwenAlignmentMismatch):
        WindowAlignment.of(
            units=["hello", "world"],
            words=[WindowAlignedWord(text="hello", start_s=0.0, end_s=0.4)],
        )


class _QwenLikeAligner:
    """A double that segments the way `transformers` really does.

    Mirrors `transformers.models.qwen3_asr.processing_qwen3_asr`: CJK
    characters are emitted individually, runs of "kept" characters (letters,
    digits, apostrophes) are emitted as one token, and everything else is
    DROPPED without ending the run, so ``black+bird`` becomes ``blackbird``.

    Modelling the real rule is the whole point: a double that tokenized on
    whitespace is what let a refusal that fires on most real CJK look green.
    The rule itself is checked against the real model by
    `scripts/prove_qwen3_fa.py`, which this cannot replace.
    """

    def __init__(self, spans: list[tuple[float, float]] | None = None) -> None:
        self.spans = spans

    @staticmethod
    def _is_cjk(char: str) -> bool:
        return 0x4E00 <= ord(char) <= 0x9FFF or 0x3400 <= ord(char) <= 0x4DBF

    @classmethod
    def _is_kept(cls, char: str) -> bool:
        import unicodedata

        if char == "'":
            return True
        category = unicodedata.category(char)
        return category.startswith("L") or category.startswith("N") or cls._is_cjk(char)

    def split_words(self, transcript: str, language: QwenLanguageLabel) -> list[str]:
        tokens: list[str] = []
        buffer: list[str] = []

        def flush() -> None:
            if buffer:
                tokens.append("".join(buffer))
                buffer.clear()

        for char in transcript.strip():
            if self._is_cjk(char):
                flush()
                tokens.append(char)
            elif char.isspace():
                flush()
            elif self._is_kept(char):
                buffer.append(char)
        flush()
        return tokens

    def align(
        self, audio: object, transcript: str, language: QwenLanguageLabel
    ) -> WindowAlignment:
        units = self.split_words(transcript, language)
        spans = self.spans
        if spans is None:
            spans = [(index * 0.1, index * 0.1 + 0.1) for index in range(len(units))]
        return WindowAlignment.of(
            units=units,
            words=[
                WindowAlignedWord(text=unit, start_s=start, end_s=end)
                for unit, (start, end) in zip(units, spans, strict=True)
            ],
        )


def _fold_intervals(words: list[str]) -> list[tuple[int, int] | None]:
    host = QwenFaHost(aligner=_QwenLikeAligner(), language=resolve_qwen_language("yue"))
    fold = host.align_words(np.zeros(SAMPLE_RATE, dtype=np.float32), words)
    return [row.interval_ms for row in fold.to_indexed_timings()]


def test_a_multi_character_cjk_word_gets_the_span_of_all_its_characters() -> None:
    """The aligner times CHARACTERS; a CHAT word is one or more of them.

    `咁搞笑` segments to three units. Its timing is the first unit's start and
    the last unit's end, which is the whole point of the fold: refusing this
    refused most real CJK, since only a character-tokenized transcript ever
    happened to match the aligner's own segmentation.
    """
    assert _fold_intervals(["咁搞笑", "嘅"]) == [(0, 300), (300, 400)]


def test_a_word_whose_characters_the_aligner_drops_folds_to_one_span() -> None:
    """`cleaned_text()` keeps `+` and non-filler `_`; the aligner keeps neither.

    So `black+bird` reaches the aligner as the single unit `blackbird`. Before
    the fold this mismatched and refused the whole group, on ordinary English.
    """
    assert _fold_intervals(["black+bird", "um_hum", "sings"]) == [
        (0, 100),
        (100, 200),
        (200, 300),
    ]


def test_a_word_with_no_alignable_characters_is_untimed_not_padded() -> None:
    """A word the aligner's tokenizer keeps nothing of owns no unit.

    It comes back UNTIMED with a named reason rather than borrowing a
    neighbour's span, which is the invented-timing failure the old refusal
    existed to prevent, now expressed per word instead of per group.
    """
    host = QwenFaHost(aligner=_QwenLikeAligner(), language=resolve_qwen_language("eng"))
    fold = host.align_words(np.zeros(SAMPLE_RATE, dtype=np.float32), ["hello", "+//"])
    assert [row.interval_ms for row in fold.to_indexed_timings()] == [(0, 100), None]
    assert fold.words[1].reason is UntimedReason.NO_ALIGNABLE_CHARACTERS


def test_a_decoded_unit_with_no_text_reads_as_blank_rather_than_raising() -> None:
    """The boundary read is TOTAL over ``text``, which is what it used to be.

    The ASR path's own guard read ``word.get("text", "") or ""`` before doing
    anything else, so a unit the decoder emitted without text was skipped. A
    later refactor replaced that with ``unit["text"]``, which turns the same
    unit into a `KeyError` in the middle of a chunk. The read is total again,
    and this pins it at the boundary rather than at one caller.
    """
    assert WindowAlignedWord.from_decoded(
        {"start_time": 0.1, "end_time": 0.4}
    ) == WindowAlignedWord(text="", start_s=0.1, end_s=0.4)


def test_a_decoded_unit_with_no_times_is_refused_by_name() -> None:
    """The times are NOT optional, and a missing one is named rather than raw.

    A unit with no times cannot be dropped: the FA fold attributes units to
    words by position, so dropping one shifts every later span onto the wrong
    word. The old code's silent skip was only ever safe because the ASR caller
    ignores position.
    """
    with pytest.raises(QwenUnitNotTimed, match="start_time"):
        WindowAlignedWord.from_decoded({"text": "hello"})


class _BlankUnitAligner:
    """An aligner that answers with a blank unit between two real ones.

    Whitespace-only units are not something `split_words_for_alignment`
    produces; this double exists because the ASR path's guard against them was
    deleted with an argument about the FA fold's arity check, which is a
    different consumer entirely.
    """

    def split_words(self, transcript: str, language: QwenLanguageLabel) -> list[str]:
        return transcript.split()

    def align(
        self, audio: object, transcript: str, language: QwenLanguageLabel
    ) -> WindowAlignment:
        words = [
            WindowAlignedWord(text="hello", start_s=0.10, end_s=0.40),
            WindowAlignedWord(text="   ", start_s=0.40, end_s=0.45),
            WindowAlignedWord(text="world", start_s=0.50, end_s=0.90),
        ]
        return WindowAlignment.of(units=[word.text for word in words], words=words)


def test_a_blank_unit_never_enters_the_asr_word_timings() -> None:
    """The ASR path reads units as WORDS, and a blank one is not a word.

    POLICY of the ASR consumer, not of the aligner: the FA fold needs every
    unit in position, so the filter cannot live in `align`. It lives in the
    transition to `SpokenWindowWords`, which is the only thing the ASR path
    reads.
    """
    recognizer = qwen_common.QwenRecognizer(
        lang="eng", model_id="Qwen/Qwen3-ASR-0.6B-hf", device="cpu"
    )
    chunk = AudioChunk(samples=np.zeros(SAMPLE_RATE, dtype=np.float32), offset=0.0)
    timings = recognizer._align_chunk(
        _loaded_qwen(_BlankUnitAligner()), chunk, "hello world"
    )
    assert timings == [(0.10, 0.40, "hello"), (0.50, 0.90, "world")]


class _DriftingAligner:
    """Segments one way when ASKED and another way when ALIGNING.

    Same unit COUNT both times, which is exactly the case the arity check
    cannot see: `prepare_forced_aligner_inputs` recomputes the segmentation
    internally, so the units the fold was proved against and the units the
    times belong to were two different calls linked only by their length.
    """

    def split_words(self, transcript: str, language: QwenLanguageLabel) -> list[str]:
        return transcript.split()

    def align(
        self, audio: object, transcript: str, language: QwenLanguageLabel
    ) -> WindowAlignment:
        drifted = ["hell", "oworld"]
        return WindowAlignment.of(
            units=drifted,
            words=[
                WindowAlignedWord(text=drifted[0], start_s=0.0, end_s=0.4),
                WindowAlignedWord(text=drifted[1], start_s=0.4, end_s=0.9),
            ],
        )


def test_a_same_count_different_boundary_segmentation_is_refused() -> None:
    host = QwenFaHost(aligner=_DriftingAligner(), language=resolve_qwen_language("eng"))
    with pytest.raises(QwenSegmentationDrift, match="hell"):
        host.align_words(np.zeros(SAMPLE_RATE, dtype=np.float32), ["hello", "world"])


class _WholeStringAligner:
    """A whole-string analyser, the shape the decomposition guard exists for.

    The Japanese and Korean branches of `split_words_for_alignment` run a
    morphological analyser over the entire string, so segmenting `"a b"` is not
    segmenting `"a"` then `"b"`. Our label map excludes them today; the
    `ForcedAligner` protocol admits any aligner, which is what makes the guard
    reachable and testable rather than decorative.
    """

    def split_words(self, transcript: str, language: QwenLanguageLabel) -> list[str]:
        return [transcript.replace(" ", "")]

    def align(
        self, audio: object, transcript: str, language: QwenLanguageLabel
    ) -> WindowAlignment:
        raise AssertionError("the segmentation check must refuse before aligning")


def test_a_whole_string_analyser_is_refused_before_any_forward_pass() -> None:
    host = QwenFaHost(
        aligner=_WholeStringAligner(), language=resolve_qwen_language("eng")
    )
    with pytest.raises(QwenSegmentationNotDecomposable, match="helloworld"):
        host.align_words(np.zeros(SAMPLE_RATE, dtype=np.float32), ["hello", "world"])
