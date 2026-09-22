"""Unit tests for _funaudio_common.py: FunAudioRecognizer logic."""

from __future__ import annotations

import builtins
import sys
import types

import pytest

from batchalign.inference.languages.cantonese._funaudio_common import (
    FunAsrSegment,
    FunAudioRecognizer,
)
from batchalign_core import BatchalignError

_S = FunAsrSegment


# ---------------------------------------------------------------------------
# FunAsrSegment.from_raw
# ---------------------------------------------------------------------------


class TestFunAsrSegmentFromRaw:
    def test_parses_normal_segment(self) -> None:
        seg = _S.from_raw({"text": "hello", "timestamp": [[0, 100]]})
        assert seg.text == "hello"
        assert seg.timestamp == [[0, 100]]

    def test_missing_text_defaults_empty(self) -> None:
        seg = _S.from_raw({})
        assert seg.text == ""
        assert seg.timestamp == []

    def test_non_list_timestamp_defaults_empty(self) -> None:
        seg = _S.from_raw({"text": "hi", "timestamp": "bad"})
        assert seg.timestamp == []

    def test_missing_timestamp_defaults_empty(self) -> None:
        seg = _S.from_raw({"text": "hi"})
        assert seg.timestamp == []

    def test_parses_checkpoint_unit_lists(self) -> None:
        paraformer = _S.from_raw(
            {"text": "好。", "timestamp": [[0, 100]], "raw_text": "好"}
        )
        assert (paraformer.raw_text, paraformer.words) == ("好", None)
        sensevoice = _S.from_raw(
            {"text": "好", "timestamp": [[0, 100]], "words": ["好"]}
        )
        assert (sensevoice.words, sensevoice.raw_text) == (["好"], None)


# ---------------------------------------------------------------------------
# transcribe, with stubbed _run_model
# ---------------------------------------------------------------------------


class TestTranscribe:
    @staticmethod
    def _make_recognizer(lang: str = "yue") -> FunAudioRecognizer:
        return FunAudioRecognizer(lang=lang)

    def test_cantonese_sensevoice_words_are_units(self) -> None:
        rec = self._make_recognizer("yue")
        rec._run_model = lambda _path: [  # type: ignore[method-assign]
            _S(
                text="<|zh|> 我仲清楚啊",
                timestamp=[[0, 100], [100, 200], [200, 300], [300, 400], [400, 500]],
                words=["我", "仲", "清", "楚", "啊"],
            )
        ]
        payload, timed_words = rec.transcribe("dummy.wav")
        values = [el["value"] for el in payload["monologues"][0]["elements"]]
        assert values == ["我", "仲", "清", "楚", "啊"]
        assert len(timed_words) == 5

    def test_cantonese_surfaces_leave_the_bridge_unnormalized(self) -> None:
        """The bridge reports what FunASR said, character for character.

        It used to normalize `真系` to `真係` here, which meant a transcript's
        text depended on which engine had produced it. Normalization now has one
        owner in the Rust server and runs once per monologue.
        """
        rec = self._make_recognizer("yue")
        rec._run_model = lambda _path: [  # type: ignore[method-assign]
            _S(text="<|zh|> 真系", timestamp=[[0, 100], [100, 200]], words=["真", "系"])
        ]
        payload, _ = rec.transcribe("dummy.wav")
        values = [el["value"] for el in payload["monologues"][0]["elements"]]
        assert values == ["真", "系"]

    def test_english_word_tokenization(self) -> None:
        rec = self._make_recognizer("eng")
        rec._run_model = lambda _path: [  # type: ignore[method-assign]
            _S(
                text="<|en|> hello world",
                timestamp=[[0, 500], [600, 1000]],
                words=["hello", "world"],
            )
        ]
        payload, timed_words = rec.transcribe("dummy.wav")
        elements = payload["monologues"][0]["elements"]
        assert [el["value"] for el in elements] == ["hello", "world"]
        assert len(timed_words) == 2

    def test_timed_words_sorted(self) -> None:
        rec = self._make_recognizer("eng")
        rec._run_model = lambda _path: [  # type: ignore[method-assign]
            _S(
                text="<|en|> hello world",
                timestamp=[[500, 800], [100, 400]],
                words=["hello", "world"],
            )
        ]
        _, timed_words = rec.transcribe("dummy.wav")
        starts = [tw["start_ms"] for tw in timed_words]
        assert starts == [100, 500]

    def test_units_and_timestamps_that_disagree_are_refused(self) -> None:
        # Three units, one timestamp. Pairing by position used to time the
        # first word and leave the rest silently untimed; admission now
        # refuses and names both counts.
        rec = self._make_recognizer("eng")
        rec._run_model = lambda _path: [  # type: ignore[method-assign]
            _S(
                text="<|en|> hello world bye",
                timestamp=[[0, 200]],
                words=["hello", "world", "bye"],
            )
        ]
        with pytest.raises(BatchalignError, match="3 units but 1 timestamps"):
            rec.transcribe("dummy.wav")

    def test_paraformer_raw_text_times_characters_not_clauses(self) -> None:
        # Paraformer timing is per character; each keeps its own span.
        rec = self._make_recognizer("zho")
        rec._run_model = lambda _path: [  # type: ignore[method-assign]
            _S(
                text="你好，世界。",
                timestamp=[[0, 100], [100, 200], [9000, 9100], [9100, 9200]],
                raw_text="你 好 世 界",
            )
        ]
        payload, timed_words = rec.transcribe("dummy.wav")
        elements = payload["monologues"][0]["elements"]
        assert [el["value"] for el in elements] == ["你", "好", "世", "界"]
        assert elements[-1]["ts"] == 9.1
        assert [tw["start_ms"] for tw in timed_words] == [0, 100, 9000, 9100]

    def test_no_timestamp_key(self) -> None:
        rec = self._make_recognizer("eng")
        rec._run_model = lambda _path: [_S(text="<|en|> hello")]  # type: ignore[method-assign]
        payload, timed_words = rec.transcribe("dummy.wav")
        assert len(payload["monologues"]) == 1
        assert payload["monologues"][0]["elements"][0]["ts"] is None
        assert timed_words == []

    def test_multiple_segments(self) -> None:
        rec = self._make_recognizer("eng")
        rec._run_model = lambda _path: [  # type: ignore[method-assign]
            _S(text="<|en|> hello", timestamp=[[0, 200]], words=["hello"]),
            _S(text="<|en|> world", timestamp=[[500, 700]], words=["world"]),
        ]
        payload, _ = rec.transcribe("dummy.wav")
        assert len(payload["monologues"]) == 2

    def test_empty_model_output(self) -> None:
        rec = self._make_recognizer("yue")
        rec._run_model = lambda _path: []  # type: ignore[method-assign]
        payload, timed_words = rec.transcribe("dummy.wav")
        assert payload["monologues"] == []
        assert timed_words == []

    def test_zero_duration_word_excluded_from_timed(self) -> None:
        rec = self._make_recognizer("eng")
        rec._run_model = lambda _path: [  # type: ignore[method-assign]
            _S(text="<|en|> hello", timestamp=[[100, 100]], words=["hello"])
        ]
        payload, timed_words = rec.transcribe("dummy.wav")
        assert len(payload["monologues"][0]["elements"]) == 1
        assert timed_words == []

    def test_speaker_is_reported_undiarized(self) -> None:
        # FunASR separates no speakers, so it names none. It used to write 0
        # here, which no downstream reader could tell apart from a provider's
        # real first speaker.
        rec = self._make_recognizer("yue")
        rec._run_model = lambda _path: [  # type: ignore[method-assign]
            _S(text="<|zh|> 好", timestamp=[[0, 100]], words=["好"])
        ]
        payload, _ = rec.transcribe("dummy.wav")
        assert payload["monologues"][0]["speaker"] == {"kind": "undiarized"}


class TestProtocolSafety:
    def test_get_model_reports_missing_dependency(self, monkeypatch) -> None:
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "funasr":
                raise ImportError("missing")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ImportError, match="funasr"):
            FunAudioRecognizer(lang="yue")._get_model()

    def test_get_model_suppresses_funasr_stdout(self, monkeypatch, capsys) -> None:
        # Verifies the ``redirect_stdout`` swallow around the
        # FunASR AutoModel construction: funasr's version banner
        # ("funasr version: ...") must not leak to the worker stdout
        # because that channel carries the V2 protocol's JSON
        # request/response stream. The recognizer's own
        # ``progress_v2`` JSON events ARE expected on stdout (they
        # ARE the protocol) and are filtered out below.
        fake_module = types.ModuleType("funasr")

        class FakeAutoModel:
            def __init__(self, **_kwargs) -> None:
                print("funasr version: 1.3.1.")

        fake_module.AutoModel = FakeAutoModel
        monkeypatch.setitem(sys.modules, "funasr", fake_module)

        rec = FunAudioRecognizer(lang="yue")
        rec._get_model()

        captured = capsys.readouterr()
        # Strip our own protocol events; what remains MUST be empty
        # (no FunASR banner, no other framework noise).
        non_protocol_lines = [
            line
            for line in captured.out.splitlines()
            if line.strip() and '"op": "progress_v2"' not in line
        ]
        assert non_protocol_lines == [], (
            f"unexpected non-protocol stdout from _get_model(): {non_protocol_lines}"
        )

    def test_get_model_uses_paraformer_constructor(self, monkeypatch) -> None:
        fake_module = types.ModuleType("funasr")
        seen: dict[str, object] = {}

        class FakeAutoModel:
            def __init__(self, **kwargs) -> None:
                seen.update(kwargs)

        fake_module.AutoModel = FakeAutoModel
        monkeypatch.setitem(sys.modules, "funasr", fake_module)

        # The auxiliary models and their revisions come from the pinned plan
        # now; they used to be five literals inside `_get_model`, which is how
        # a model could move without anything recording that it had.
        rec = FunAudioRecognizer(
            lang="yue",
            model="paraformer-zh",
            device="cuda",
            model_revision="v2.0.4",
            vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            vad_revision="v2.0.4",
            punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            punc_revision="v2.0.4",
        )
        rec._get_model()

        assert seen["model"] == "paraformer-zh"
        assert seen["model_revision"] == "v2.0.4"
        assert seen["vad_model"] == "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
        assert seen["vad_model_revision"] == "v2.0.4"
        assert (
            seen["punc_model"]
            == "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
        )
        assert seen["punc_model_revision"] == "v2.0.4"

    def test_get_model_refuses_paraformer_without_its_auxiliary_models(self) -> None:
        """The Paraformer branch needs its voice-activity and punctuation
        models: loading with FunASR's own defaults would silently run models
        nothing recorded."""
        rec = FunAudioRecognizer(lang="yue", model="paraformer-zh")
        with pytest.raises(ValueError, match="voice-activity and punctuation"):
            rec._get_model()

    def test_run_model_suppresses_generate_stdout(self, capsys) -> None:
        class FakeModel:
            def generate(self, **_kwargs):
                print("funasr version: 1.3.1.")
                return [{"text": "<|zh|> 好", "timestamp": [[0, 100]]}]

        rec = FunAudioRecognizer(lang="yue")
        rec._model = FakeModel()

        segments = rec._run_model("dummy.wav")

        captured = capsys.readouterr()
        assert captured.out == ""
        assert segments == [_S(text="<|zh|> 好", timestamp=[[0, 100]])]

    def test_run_model_accepts_paraformer_dict_output(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                return {"text": "好。", "raw_text": "好", "timestamp": [[0, 100]]}

        rec = FunAudioRecognizer(lang="zho", model="paraformer-zh")
        rec._model = fake_model = FakeModel()

        segments = rec._run_model("dummy.wav")

        # `return_raw_text` is what makes the timing pairable; see FunAsrSegment.
        assert fake_model.calls == [
            {"input": "dummy.wav", "output_timestamp": True, "return_raw_text": True}
        ]
        assert segments == [_S(text="好。", timestamp=[[0, 100]], raw_text="好")]

    def test_run_model_ignores_non_collection_output(self) -> None:
        class FakeModel:
            def generate(self, **_kwargs):
                return "not-a-segment-list"

        rec = FunAudioRecognizer(lang="yue")
        rec._model = FakeModel()

        assert rec._run_model("dummy.wav") == []


class TestFunaudioDownloadEvents:
    """Time-transparency: FunASR's first-use model load must emit
    ``progress_v2`` download events so the daemon log / dashboard /
    TUI show "Loading FunASR model…" instead of dead air during the
    multi-minute first-time HuggingFace download.

    Per AGENTS.md §11 (talkbank-tools/AGENTS.md), every operation that
    takes more than ~1 second must surface to all UI channels via
    ``emit_download_event``. FunASR's lazy ``_get_model()`` was an
    observability gap: this test pins the start/complete event pair
    on cache miss and the no-event behavior on cache hit.
    """

    def test_first_get_model_emits_start_and_complete_events(self, monkeypatch) -> None:
        import sys

        from batchalign.inference.languages.cantonese._funaudio_common import (
            FunAudioRecognizer,
        )

        # Fake ``funasr.AutoModel`` so the test does not actually
        # download or load a ~1 GB model from HuggingFace.
        class _FakeAutoModel:
            def __init__(self, **_kwargs):
                pass

        fake_funasr_mod = type(sys)("fake_funasr_mod")
        fake_funasr_mod.AutoModel = _FakeAutoModel  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "funasr", fake_funasr_mod)

        # Capture every ``emit_download_event`` call so the test can
        # assert on stage names + ordering.
        events: list[dict[str, str]] = []

        def fake_emit(stage: str, user_message: str, **_kwargs) -> None:
            events.append({"stage": stage, "user_message": user_message})

        monkeypatch.setattr(
            "batchalign.worker._progress.emit_download_event", fake_emit
        )

        rec = FunAudioRecognizer(lang="yue")
        rec._get_model()

        assert len(events) == 2, (
            f"first _get_model() must emit start+complete events; "
            f"got {len(events)} event(s): {events}"
        )
        assert events[0]["stage"] == "downloading_funaudio_asr"
        assert (
            "FunASR" in events[0]["user_message"]
            or "FunAudio" in events[0]["user_message"]
        )
        assert events[1]["stage"] == "downloading_funaudio_asr_complete"

    def test_second_get_model_call_is_cache_hit_no_events(self, monkeypatch) -> None:
        # Once the model is loaded, subsequent ``_get_model()`` calls
        # MUST NOT re-emit the download events; they would mislead
        # the user into thinking another download is happening.
        import sys

        from batchalign.inference.languages.cantonese._funaudio_common import (
            FunAudioRecognizer,
        )

        class _FakeAutoModel:
            def __init__(self, **_kwargs):
                pass

        fake_funasr_mod = type(sys)("fake_funasr_mod")
        fake_funasr_mod.AutoModel = _FakeAutoModel  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "funasr", fake_funasr_mod)

        events: list[dict[str, str]] = []

        def fake_emit(stage: str, user_message: str, **_kwargs) -> None:
            events.append({"stage": stage, "user_message": user_message})

        monkeypatch.setattr(
            "batchalign.worker._progress.emit_download_event", fake_emit
        )

        rec = FunAudioRecognizer(lang="yue")
        rec._get_model()  # First call: 2 events.
        events.clear()
        rec._get_model()  # Second call: cache hit, 0 events.
        rec._get_model()  # Third call: same.

        assert events == [], (
            f"subsequent _get_model() calls must hit cache without "
            f"emitting download events; got {events}"
        )
