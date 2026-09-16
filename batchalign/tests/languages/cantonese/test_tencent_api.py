"""Unit tests for _tencent_api.py: TencentRecognizer monologues/timed_words logic."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from batchalign.inference.languages.cantonese._tencent_api import TencentRecognizer


def _make_recognizer(lang: str = "yue") -> TencentRecognizer:
    """Create a TencentRecognizer without calling __init__ (skips credential loading)."""
    rec = TencentRecognizer.__new__(TencentRecognizer)
    rec.lang_code = lang
    # Carried in by the control plane now; these tests exercise the projection
    # helpers, which read only `lang_code`.
    rec.engine_model_type = "16k_zh_large"
    return rec


# The engine-model-type tests that lived here are gone with the derivation
# they covered. Choosing that value is now the Rust control plane's job
# (`model_manifest::tencent_engine_model_type`), and it is tested there:
# `every_chinese_variety_including_mandarin_asks_for_the_chinese_model` and
# `languages_with_a_two_letter_code_keep_their_model_type`. Those also cover
# `cmn`, which the tests here never did, and which is why Mandarin used to ask
# Tencent for a `16k_cmn` model that does not exist.


# ---------------------------------------------------------------------------
# monologues: using real-style data from 05b.cha
# ---------------------------------------------------------------------------


class TestMonologues:
    def test_basic_cantonese(self) -> None:
        rec = _make_recognizer("yue")
        result_detail = [
            SimpleNamespace(
                StartMs=4850,
                SpeakerId=1,
                Words=[
                    SimpleNamespace(Word="咁", OffsetStartMs=0, OffsetEndMs=200),
                    SimpleNamespace(Word="搞", OffsetStartMs=250, OffsetEndMs=500),
                    SimpleNamespace(Word="笑", OffsetStartMs=550, OffsetEndMs=800),
                    SimpleNamespace(Word="嘅", OffsetStartMs=850, OffsetEndMs=1025),
                ],
            ),
        ]
        payload = rec.monologues(result_detail)
        assert len(payload["monologues"]) == 1
        elements = payload["monologues"][0]["elements"]
        assert len(elements) == 4
        assert elements[0]["value"] == "咁"
        assert abs(elements[0]["ts"] - 4.850) < 0.001
        assert abs(elements[0]["end_ts"] - 5.050) < 0.001

    def test_no_normalization_is_applied_here(self) -> None:
        """Monologue surfaces are Tencent's own.

        This asserted `系` becoming `係` until 2026-09-16. Normalization has one
        owner now, `AlignedNormalization` in the Rust server, which runs once
        per monologue after the provider payload arrives.
        """
        rec = _make_recognizer("yue")
        result_detail = [
            SimpleNamespace(
                StartMs=1000,
                SpeakerId=1,
                Words=[
                    SimpleNamespace(Word="系", OffsetStartMs=0, OffsetEndMs=200),
                ],
            ),
        ]
        payload = rec.monologues(result_detail)
        assert payload["monologues"][0]["elements"][0]["value"] == "系"

    def test_empty_result_detail(self) -> None:
        rec = _make_recognizer("yue")
        payload = rec.monologues([])
        assert payload["monologues"] == []

    def test_empty_words_skipped(self) -> None:
        rec = _make_recognizer("yue")
        result_detail = [SimpleNamespace(StartMs=0, SpeakerId=1, Words=[])]
        payload = rec.monologues(result_detail)
        assert payload["monologues"] == []

    def test_blank_word_skipped(self) -> None:
        rec = _make_recognizer("yue")
        result_detail = [
            SimpleNamespace(
                StartMs=0,
                SpeakerId=1,
                Words=[
                    SimpleNamespace(Word="  ", OffsetStartMs=0, OffsetEndMs=100),
                    SimpleNamespace(Word="好", OffsetStartMs=200, OffsetEndMs=300),
                ],
            ),
        ]
        payload = rec.monologues(result_detail)
        elements = payload["monologues"][0]["elements"]
        assert len(elements) == 1
        assert elements[0]["value"] == "好"

    def test_multi_speaker(self) -> None:
        rec = _make_recognizer("yue")
        result_detail = [
            SimpleNamespace(
                StartMs=6750,
                SpeakerId=1,
                Words=[
                    SimpleNamespace(Word="我", OffsetStartMs=0, OffsetEndMs=150),
                    SimpleNamespace(Word="仲", OffsetStartMs=200, OffsetEndMs=350),
                ],
            ),
            SimpleNamespace(
                StartMs=8930,
                SpeakerId=2,
                Words=[
                    SimpleNamespace(Word="我", OffsetStartMs=0, OffsetEndMs=200),
                ],
            ),
        ]
        payload = rec.monologues(result_detail)
        assert len(payload["monologues"]) == 2
        assert payload["monologues"][0]["speaker"] == {
            "kind": "attributed",
            "label": "1",
        }
        assert payload["monologues"][1]["speaker"] == {
            "kind": "attributed",
            "label": "2",
        }

    def test_missing_attributes_handled(self) -> None:
        rec = _make_recognizer("yue")
        result_detail = [SimpleNamespace()]
        payload = rec.monologues(result_detail)
        assert payload["monologues"] == []


class TestAbsentFieldAdmission:
    """Absent Tencent fields are STATES, never zeros.

    Every property of Tencent's `SentenceDetail` and `SentenceWords` is
    documented as nullable, and the SDK initializes each attribute to `None`
    before deserializing a response. The projection used to read them with
    `.unwrap_or(0)`, so a word with no offsets claimed to be spoken at the start
    of its segment and a segment with no `StartMs` claimed to start at the
    beginning of the recording. Both fabrications reached CHAT.
    """

    def test_a_segment_without_a_start_yields_untimed_words_not_zero_based_ones(
        self,
    ) -> None:
        rec = _make_recognizer("yue")
        result_detail = [
            SimpleNamespace(
                SpeakerId=1,
                Words=[
                    SimpleNamespace(Word="好", OffsetStartMs=0, OffsetEndMs=200),
                    SimpleNamespace(Word="嗎", OffsetStartMs=300, OffsetEndMs=500),
                ],
            ),
        ]

        payload = rec.monologues(result_detail)
        elements = payload["monologues"][0]["elements"]

        assert [element["value"] for element in elements] == ["好", "嗎"]
        assert all(element["ts"] is None for element in elements)
        assert all(element["end_ts"] is None for element in elements)
        assert rec.timed_words(result_detail) == []

    def test_a_word_without_offsets_is_untimed_while_its_neighbours_keep_timing(
        self,
    ) -> None:
        rec = _make_recognizer("yue")
        result_detail = [
            SimpleNamespace(
                StartMs=4850,
                SpeakerId=1,
                Words=[
                    SimpleNamespace(Word="好"),
                    SimpleNamespace(Word="嗎", OffsetStartMs=300, OffsetEndMs=500),
                ],
            ),
        ]

        elements = rec.monologues(result_detail)["monologues"][0]["elements"]

        assert elements[0]["ts"] is None and elements[0]["end_ts"] is None
        assert abs(elements[1]["ts"] - 5.150) < 0.001
        timed = rec.timed_words(result_detail)
        assert [word["word"] for word in timed] == ["嗎"]

    def test_a_word_with_only_one_offset_is_untimed(self) -> None:
        # The end used to default to the start, producing a zero-width span
        # that claimed a position it had not been given.
        rec = _make_recognizer("yue")
        result_detail = [
            SimpleNamespace(
                StartMs=1000,
                SpeakerId=1,
                Words=[SimpleNamespace(Word="好", OffsetStartMs=100)],
            ),
        ]

        elements = rec.monologues(result_detail)["monologues"][0]["elements"]
        assert elements[0]["ts"] is None
        assert elements[0]["end_ts"] is None

    def test_a_word_without_a_surface_refuses_instead_of_vanishing(self) -> None:
        # `unwrap_or_default()` made this an empty string, which the blank
        # filter then dropped: a word Tencent failed to send disappeared with
        # no trace at all.
        rec = _make_recognizer("yue")
        result_detail = [
            SimpleNamespace(
                StartMs=0,
                SpeakerId=1,
                Words=[SimpleNamespace(OffsetStartMs=0, OffsetEndMs=100)],
            ),
        ]

        with pytest.raises(Exception) as refusal:
            rec.monologues(result_detail)
        message = str(refusal.value)
        assert "Tencent" in message, message
        assert "segment 0 word 0" in message, message
        assert "Word" in message, message

    def test_a_wrongly_typed_time_refuses_rather_than_reading_as_zero(self) -> None:
        rec = _make_recognizer("yue")
        result_detail = [
            SimpleNamespace(
                StartMs="not a number",
                SpeakerId=1,
                Words=[SimpleNamespace(Word="好", OffsetStartMs=0, OffsetEndMs=100)],
            ),
        ]

        with pytest.raises(Exception) as refusal:
            rec.monologues(result_detail)
        message = str(refusal.value)
        assert "segment 0" in message, message
        assert "StartMs" in message, message

    def test_an_inverted_word_span_is_refused_by_name(self) -> None:
        rec = _make_recognizer("yue")
        result_detail = [
            SimpleNamespace(
                StartMs=1000,
                SpeakerId=1,
                Words=[SimpleNamespace(Word="好", OffsetStartMs=500, OffsetEndMs=100)],
            ),
        ]

        with pytest.raises(Exception) as refusal:
            rec.monologues(result_detail)
        assert "before its start" in str(refusal.value)

    def test_a_segment_without_a_speaker_is_undiarized(self) -> None:
        # An absent `SpeakerId` means the provider named nobody, and that is
        # now what the adapter reports. It no longer becomes track 0, which was
        # indistinguishable from a real first speaker. Adjudicating that
        # against what the request asked for belongs to the worker bridge,
        # which refuses an unnamed speaker when it asked for separation.
        rec = _make_recognizer("yue")
        result_detail = [
            SimpleNamespace(
                StartMs=0,
                Words=[SimpleNamespace(Word="好", OffsetStartMs=0, OffsetEndMs=100)],
            ),
        ]

        payload = rec.monologues(result_detail)
        assert payload["monologues"][0]["speaker"] == {"kind": "undiarized"}

    def test_english_no_normalization(self) -> None:
        rec = _make_recognizer("eng")
        rec.engine_model_type = "16k_en"
        result_detail = [
            SimpleNamespace(
                StartMs=0,
                SpeakerId=1,
                Words=[
                    SimpleNamespace(Word="hello", OffsetStartMs=0, OffsetEndMs=500),
                ],
            ),
        ]
        payload = rec.monologues(result_detail)
        assert payload["monologues"][0]["elements"][0]["value"] == "hello"


# ---------------------------------------------------------------------------
# timed_words
# ---------------------------------------------------------------------------


class TestTimedWords:
    def test_basic(self) -> None:
        rec = _make_recognizer("yue")
        result_detail = [
            SimpleNamespace(
                StartMs=1000,
                SpeakerId=1,
                Words=[
                    SimpleNamespace(Word="好", OffsetStartMs=0, OffsetEndMs=200),
                    SimpleNamespace(Word="似", OffsetStartMs=300, OffsetEndMs=450),
                ],
            ),
        ]
        timed = rec.timed_words(result_detail)
        assert len(timed) == 2
        assert timed[0]["word"] == "好"
        assert timed[0]["start_ms"] == 1000
        assert timed[0]["end_ms"] == 1200
        assert timed[1]["word"] == "似"
        assert timed[1]["start_ms"] == 1300
        assert timed[1]["end_ms"] == 1450

    def test_zero_duration_filtered(self) -> None:
        rec = _make_recognizer("yue")
        result_detail = [
            SimpleNamespace(
                StartMs=0,
                SpeakerId=1,
                Words=[
                    SimpleNamespace(Word="嗯", OffsetStartMs=100, OffsetEndMs=100),
                ],
            ),
        ]
        assert rec.timed_words(result_detail) == []

    def test_sorted_across_segments(self) -> None:
        rec = _make_recognizer("yue")
        result_detail = [
            SimpleNamespace(
                StartMs=5000,
                SpeakerId=1,
                Words=[
                    SimpleNamespace(Word="後", OffsetStartMs=0, OffsetEndMs=200),
                ],
            ),
            SimpleNamespace(
                StartMs=1000,
                SpeakerId=1,
                Words=[
                    SimpleNamespace(Word="前", OffsetStartMs=0, OffsetEndMs=200),
                ],
            ),
        ]
        timed = rec.timed_words(result_detail)
        assert timed[0]["word"] == "前"
        assert timed[1]["word"] == "後"

    def test_timed_words_keep_tencents_own_characters(self) -> None:
        """Tencent's surfaces cross the bridge unchanged.

        `呀` used to arrive as `啊` because the bridge normalized every token.
        Normalization now happens once, in the server, for every engine.
        """
        rec = _make_recognizer("yue")
        result_detail = [
            SimpleNamespace(
                StartMs=500,
                SpeakerId=1,
                Words=[
                    SimpleNamespace(Word="呀", OffsetStartMs=0, OffsetEndMs=100),
                ],
            ),
        ]
        timed = rec.timed_words(result_detail)
        assert timed[0]["word"] == "呀"

    def test_empty_input(self) -> None:
        rec = _make_recognizer("yue")
        assert rec.timed_words([]) == []
