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

import logging

import numpy as np
import pytest

from batchalign.inference.languages.cantonese import _qwen_common as qwen_common
from batchalign.inference.languages.cantonese._qwen_chunking import (
    SAMPLE_RATE,
    AudioChunk,
)
from batchalign.inference.qwen_forced_alignment import (
    BackwardsUnitSpan,
    IndexedWindowUnit,
    InvalidUnitSpan,
    MissingUnitTime,
    PairedWindowAlignment,
    QwenAlignmentMismatch,
    QwenFaHost,
    QwenLanguageLabel,
    QwenSegmentationDrift,
    QwenSegmentationNotDecomposable,
    SpanDefect,
    TimeField,
    UntimedChatWord,
    UntimedReason,
    UntimedReportedUnit,
    WindowAlignedWord,
    read_decoded_unit,
    resolve_qwen_language,
)
from batchalign.inference.qwen_timing import (
    AudioWindow,
    DisorderedWindowTiming,
    ObservedRecordingSpan,
    ObservedWindowSpan,
    QwenTimingInvalid,
    RejectedWindowTiming,
    UnavailableWindowTiming,
    WindowAlignment,
)
from batchalign.tests._qwen_recognition import SEGMENTATION_MODEL_POLICY


def _admit(units, words, duration_s=1.0):
    return WindowAlignment(
        PairedWindowAlignment.of(units, words), AudioWindow(duration_s)
    )


def _rows(chunk):
    result = []
    for word in chunk.words:
        assert isinstance(word.timing, ObservedRecordingSpan)
        result.append((word.timing.start_s, word.timing.end_s, word.text))
    return result


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
    ) -> PairedWindowAlignment:
        self.calls.append((transcript, str(language)))
        return PairedWindowAlignment.of(
            units=[word.text for word in self.aligned], words=self.aligned
        )


def _loaded_qwen(aligner: _RecordingAligner) -> qwen_common.LoadedQwen:
    """A ``LoadedQwen`` whose ASR half is inert and whose aligner is the fake."""
    return qwen_common.LoadedQwen(processor=object(), model=object(), aligner=aligner)


def _ready_qwen(aligner):
    return qwen_common.ReadyQwenRecognizer(
        qwen_common.QwenRecognizerConfig(
            lang="eng",
            device="cpu",
            recognition=SEGMENTATION_MODEL_POLICY,
        ),
        _loaded_qwen(aligner),
    )


def test_both_callers_reach_the_aligner_through_the_same_seam() -> None:
    aligner = _RecordingAligner(
        [
            WindowAlignedWord(text="hello", start_s=0.10, end_s=0.40),
            WindowAlignedWord(text="world", start_s=0.50, end_s=0.90),
        ]
    )

    recognizer = _ready_qwen(aligner)
    chunk = AudioChunk(samples=np.zeros(SAMPLE_RATE, dtype=np.float32), offset=0.0)
    recognizer._align_chunk(chunk, "hello world")

    host = QwenFaHost(aligner=aligner, language=resolve_qwen_language("eng"))
    host.align_words(np.zeros(SAMPLE_RATE, dtype=np.float32), ["hello", "world"])

    assert [transcript for transcript, _ in aligner.calls] == [
        "hello world",
        "hello world",
    ]


def test_only_the_asr_chunk_path_converts_window_time_to_file_time() -> None:
    aligned = [WindowAlignedWord(text="hello", start_s=0.10, end_s=0.40)]

    recognizer = _ready_qwen(_RecordingAligner(aligned))
    chunk = AudioChunk(samples=np.zeros(SAMPLE_RATE, dtype=np.float32), offset=12.0)
    asr_timings = recognizer._align_chunk(chunk, "hello")
    assert _rows(asr_timings) == [(12.10, 12.40, "hello")]

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
        PairedWindowAlignment.of(
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
    ) -> PairedWindowAlignment:
        units = self.split_words(transcript, language)
        spans = self.spans
        if spans is None:
            spans = [(index * 0.1, index * 0.1 + 0.1) for index in range(len(units))]
        return PairedWindowAlignment.of(
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
    untimed = fold.words[1]
    assert isinstance(untimed, UntimedChatWord)
    assert untimed.reason is UntimedReason.NO_ALIGNABLE_CHARACTERS


def test_a_decoded_unit_with_no_text_reads_as_blank_rather_than_raising() -> None:
    """The boundary read is TOTAL over ``text``, which is what it used to be.

    The ASR path's own guard read ``word.get("text", "") or ""`` before doing
    anything else, so a unit the decoder emitted without text was skipped. A
    later refactor replaced that with ``unit["text"]``, which turns the same
    unit into a `KeyError` in the middle of a chunk. The read is total again,
    and this pins it at the boundary rather than at one caller.
    """
    assert read_decoded_unit({"start_time": 0.1, "end_time": 0.4}) == WindowAlignedWord(
        text="", start_s=0.1, end_s=0.4
    )


@pytest.mark.parametrize(
    "decoded,field",
    [
        ({"text": "hello"}, TimeField.START),
        ({"text": "hello", "start_time": 0.1}, TimeField.END),
        ({"text": "hello", "start_time": "soon", "end_time": 0.4}, TimeField.START),
        ({"text": "hello", "start_time": 0.1, "end_time": True}, TimeField.END),
    ],
)
def test_a_decoded_unit_without_a_usable_time_keeps_its_position_untimed(
    decoded: dict[str, object], field: TimeField
) -> None:
    """A missing or non-numeric time is neither dropped (the fold attributes
    units to words by position) nor invented: the unit is untimed, the file
    goes on, and its neighbours keep their timing."""
    reported = read_decoded_unit(decoded)
    assert isinstance(reported, UntimedReportedUnit)
    assert (reported.text, reported.field) == ("hello", field)
    alignment = WindowAlignment(
        PairedWindowAlignment.of(
            ["hello", "world"], [reported, WindowAlignedWord("world", 0.5, 0.6)]
        ),
        AudioWindow(1.0),
    )
    missing = alignment.words[0].timing
    assert isinstance(missing, RejectedWindowTiming)
    assert isinstance(missing.defect, MissingUnitTime)
    assert missing.defect.index == 0
    assert isinstance(alignment.words[1].timing, ObservedWindowSpan)


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
    ) -> PairedWindowAlignment:
        words = [
            WindowAlignedWord(text="hello", start_s=0.10, end_s=0.40),
            WindowAlignedWord(text="   ", start_s=0.40, end_s=0.45),
            WindowAlignedWord(text="world", start_s=0.50, end_s=0.90),
        ]
        return PairedWindowAlignment.of(
            units=[word.text for word in words], words=words
        )


def test_a_blank_unit_never_enters_the_asr_word_timings() -> None:
    """The ASR path reads units as WORDS, and a blank one is not a word.

    POLICY of the ASR consumer, not of the aligner: the FA fold needs every
    unit in position, so the filter cannot live in `align`. It lives in the
    transition to `SpokenWindowWords`, which is the only thing the ASR path
    reads.
    """
    recognizer = _ready_qwen(_BlankUnitAligner())
    chunk = AudioChunk(samples=np.zeros(SAMPLE_RATE, dtype=np.float32), offset=0.0)
    timings = recognizer._align_chunk(chunk, "hello world")
    assert _rows(timings) == [(0.10, 0.40, "hello"), (0.50, 0.90, "world")]


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
    ) -> PairedWindowAlignment:
        drifted = ["hell", "oworld"]
        return PairedWindowAlignment.of(
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
    ) -> PairedWindowAlignment:
        raise AssertionError("the segmentation check must refuse before aligning")


def test_a_whole_string_analyser_is_refused_before_any_forward_pass() -> None:
    host = QwenFaHost(
        aligner=_WholeStringAligner(), language=resolve_qwen_language("eng")
    )
    with pytest.raises(QwenSegmentationNotDecomposable, match="helloworld"):
        host.align_words(np.zeros(SAMPLE_RATE, dtype=np.float32), ["hello", "world"])


@pytest.mark.parametrize(
    "start,end,defect",
    [
        (float("nan"), 1.0, SpanDefect.NON_FINITE),
        (0.0, float("inf"), SpanDefect.NON_FINITE),
        (-0.1, 0.5, SpanDefect.NEGATIVE),
        (0.7, 0.2, SpanDefect.REVERSED),
    ],
)
def test_a_unit_with_an_invalid_span_is_untimed_keeping_its_defect(
    start: float, end: float, defect: SpanDefect
) -> None:
    """One unit's unusable span untimes that unit; it never fails the window."""
    alignment = _admit(
        ["hello", "world"],
        [WindowAlignedWord("hello", start, end), WindowAlignedWord("world", 0.8, 0.9)],
    )
    rejected = alignment.words[0].timing
    assert isinstance(rejected, RejectedWindowTiming)
    assert isinstance(rejected.defect, InvalidUnitSpan)
    assert rejected.defect.defect is defect
    assert isinstance(alignment.words[1].timing, ObservedWindowSpan)


@pytest.mark.parametrize(
    "previous,current,boundary",
    [
        ((0.7, 0.8), (0.5, 0.6), "start"),
        ((0.1, 0.8), (0.5, 0.6), "end"),
    ],
)
def test_a_backwards_pair_untimes_its_window_keeping_both_units(
    previous: tuple[float, float], current: tuple[float, float], boundary: str
) -> None:
    """The window's order is untrustworthy, so none of its spans is kept, and
    nothing raises: the file keeps going."""
    alignment = _admit(
        ["hello", "world", "again"],
        [
            WindowAlignedWord("hello", *previous),
            WindowAlignedWord("world", *current),
            WindowAlignedWord("again", 0.9, 0.95),
        ],
    )
    timings = [word.timing for word in alignment.words]
    assert all(isinstance(timing, DisorderedWindowTiming) for timing in timings)
    defect = timings[0].defect
    assert isinstance(defect, BackwardsUnitSpan)
    assert all(timing.defect == defect for timing in timings)
    assert (defect.previous.index, defect.current.index) == (0, 1)
    assert defect.boundary.value == boundary


def test_a_unit_defect_keeps_its_own_reason_in_a_disordered_window() -> None:
    alignment = _admit(
        ["hello", "world", "late"],
        [
            WindowAlignedWord("hello", 0.7, 0.8),
            WindowAlignedWord("world", 0.5, 0.6),
            WindowAlignedWord("late", 0.9, 1.5),
        ],
    )
    assert isinstance(alignment.words[0].timing, DisorderedWindowTiming)
    assert isinstance(alignment.words[2].timing, RejectedWindowTiming)


def test_a_span_constructor_cannot_be_built_from_an_unapproved_span() -> None:
    """Admission never raises; the constructors that claim a usable span do."""
    unit = IndexedWindowUnit(0, WindowAlignedWord("hello", 0.5, 1.5))
    with pytest.raises(QwenTimingInvalid):
        ObservedWindowSpan(unit, AudioWindow(1.0))


def test_the_fa_fold_leaves_a_disordered_window_untimed_without_failing() -> None:
    host = QwenFaHost(
        aligner=_RecordingAligner(
            [WindowAlignedWord("hello", 0.7, 0.8), WindowAlignedWord("world", 0.5, 0.6)]
        ),
        language=resolve_qwen_language("eng"),
    )
    fold = host.align_words(np.zeros(SAMPLE_RATE, dtype=np.float32), ["hello", "world"])
    assert [row.interval_ms for row in fold.to_indexed_timings()] == [None, None]
    assert all(
        isinstance(word, UntimedChatWord)
        and word.reason is UntimedReason.ALIGNMENT_UNIT_UNAVAILABLE
        for word in fold.words
    )


def test_asr_untimes_a_disordered_chunk_and_logs_its_recording_window(
    caplog: pytest.LogCaptureFixture,
) -> None:
    recognizer = _ready_qwen(
        _RecordingAligner(
            [WindowAlignedWord("hello", 0.7, 0.8), WindowAlignedWord("world", 0.5, 0.6)]
        )
    )
    chunk = AudioChunk(samples=np.zeros(SAMPLE_RATE), offset=180.0)
    with caplog.at_level(logging.WARNING):
        speech = recognizer._align_chunk(chunk, "hello world")
    assert [word.text for word in speech.words] == ["hello", "world"]
    assert all(isinstance(word.timing, DisorderedWindowTiming) for word in speech.words)
    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "recording window 180..181 s" in logged
    assert "relative to that window" in logged
    assert "0.5..0.6" in logged
    assert "the rest of the file keeps its timing" in logged


def test_blank_and_zero_width_units_keep_their_fold_positions() -> None:
    words = (
        WindowAlignedWord("hello", 0.1, 0.4),
        WindowAlignedWord("", 0.0, 0.0),
        WindowAlignedWord("unknown", 0.0, 0.0),
        WindowAlignedWord("world", 0.3, 0.7),
    )
    alignment = _admit([word.text for word in words], words)
    assert [word.text for word in alignment.words] == [word.text for word in words]
    assert isinstance(alignment.words[1].timing, UnavailableWindowTiming)
    assert isinstance(alignment.words[2].timing, UnavailableWindowTiming)
