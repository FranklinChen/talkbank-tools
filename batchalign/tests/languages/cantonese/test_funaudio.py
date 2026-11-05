"""Unit tests for _funaudio_common.py: FunAudioRecognizer logic."""

from __future__ import annotations

import builtins
import sys
import types

import pytest

from batchalign.inference.languages.cantonese._funaudio_common import (
    FunAsrSegment,
    FunAsrSnapshots,
    FunAudioRecognizer,
    ParaformerSnapshots,
    SenseVoiceSnapshots,
)
from batchalign.tests._snapshots import modelscope_snapshot, repository_snapshot
from batchalign.worker._model_errors import ModelOutputInvalidError
from batchalign.worker._types_v2 import PhaseEdgeV2, PhaseV2

_S = FunAsrSegment

SENSEVOICE = SenseVoiceSnapshots(
    asr=repository_snapshot("org/sensevoice"),
    vad=repository_snapshot("org/vad"),
)
PARAFORMER = ParaformerSnapshots(
    asr=modelscope_snapshot("iic/paraformer"),
    vad=modelscope_snapshot("iic/vad"),
    punc=modelscope_snapshot("iic/punc"),
)


def recognizer(
    model: object = None, lang: str = "yue", snapshots: FunAsrSnapshots = SENSEVOICE
) -> FunAudioRecognizer:
    """A recognizer around a stand-in model, as ``load`` would return one."""
    return FunAudioRecognizer(
        lang=lang, device="cpu", snapshots=snapshots, label="test", model=model
    )


def fake_funasr(monkeypatch, auto_model: type) -> None:
    """Install ``auto_model`` as ``funasr.AutoModel`` for one test."""
    module = types.ModuleType("funasr")
    module.AutoModel = auto_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "funasr", module)


# ---------------------------------------------------------------------------
# FunAsrSegment.from_raw
# ---------------------------------------------------------------------------


class TestFunAsrSegmentFromRaw:
    def test_parses_normal_segment(self) -> None:
        seg = _S.from_raw({"text": "hello", "timestamp": [[0, 100]]})
        assert seg.text == "hello"
        assert seg.timestamp == [[0, 100]]

    def test_missing_text_is_not_fabricated_as_silence(self) -> None:
        with pytest.raises(ModelOutputInvalidError, match="text field"):
            _S.from_raw({})

    @pytest.mark.parametrize(
        ("field", "value"),
        [("timestamp", "bad"), ("words", ["hi", 1]), ("raw_text", 42)],
    )
    def test_malformed_fields_are_not_discarded(self, field, value) -> None:
        with pytest.raises(ModelOutputInvalidError, match=field):
            _S.from_raw({"text": "hi", field: value})

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
        return recognizer(lang=lang)

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
        with pytest.raises(ModelOutputInvalidError, match="3 units but 1 timestamps"):
            rec.transcribe("dummy.wav")

    def test_tokenizer_markers_are_decoded_without_shifting_word_times(self) -> None:
        rec = self._make_recognizer("eng")
        rec._run_model = lambda _path: [  # type: ignore[method-assign]
            _S(
                text="hello world",
                words=["▁hello", "▁", "<|en|>▁world"],
                timestamp=[[100, 200], [200, 300], [300, 400]],
            )
        ]
        payload, timed_words = rec.transcribe("dummy.wav")
        elements = payload["monologues"][0]["elements"]
        assert [(el["value"], el["ts"]) for el in elements] == [
            ("hello", 0.1),
            ("world", 0.3),
        ]
        assert [word["start_ms"] for word in timed_words] == [100, 300]

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
    def test_native_boundary_refuses_missing_text_too(self) -> None:
        import batchalign_core

        with pytest.raises(ModelOutputInvalidError, match="text"):
            batchalign_core.funaudio_segments_to_asr([{}])

    def test_load_reports_missing_dependency(self, monkeypatch) -> None:
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "funasr":
                raise ImportError("missing")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ImportError, match="funasr"):
            FunAudioRecognizer.load(SENSEVOICE, lang="yue", device="cpu", label="t")

    def test_load_suppresses_funasr_stdout(self, monkeypatch, capsys) -> None:
        # FunASR's version banner must not reach the worker's stdout, which
        # carries the protocol. The recognizer's own `progress_v2` events ARE
        # protocol and are filtered out below.
        class FakeAutoModel:
            def __init__(self, **_kwargs) -> None:
                print("funasr version: 1.3.1.")

        fake_funasr(monkeypatch, FakeAutoModel)

        FunAudioRecognizer.load(SENSEVOICE, lang="yue", device="cpu", label="t")

        non_protocol_lines = [
            line
            for line in capsys.readouterr().out.splitlines()
            if line.strip() and '"op": "progress_v2"' not in line
        ]
        assert non_protocol_lines == []

    @pytest.mark.parametrize(
        ("snapshots", "expected"),
        [
            (
                SENSEVOICE,
                {"model": SENSEVOICE.asr.path, "vad_model": SENSEVOICE.vad.path},
            ),
            (
                PARAFORMER,
                {
                    "model": str(PARAFORMER.asr.path),
                    "vad_model": str(PARAFORMER.vad.path),
                    "punc_model": str(PARAFORMER.punc.path),
                },
            ),
        ],
    )
    def test_load_builds_each_composition_from_local_paths_only(
        self, monkeypatch, snapshots, expected
    ) -> None:
        """No model id and no revision reaches FunASR, so it has nothing to
        resolve against a hub: the worker materialized every member first."""
        seen: dict[str, object] = {}

        class FakeAutoModel:
            def __init__(self, **kwargs) -> None:
                seen.update(kwargs)

        fake_funasr(monkeypatch, FakeAutoModel)

        loaded = FunAudioRecognizer.load(
            snapshots, lang="yue", device="cuda", label="t"
        )

        assert {key: seen[key] for key in expected} == expected
        assert seen["device"] == "cuda"
        assert seen["disable_update"] is True
        assert not {key for key in seen if key.endswith("revision")}
        assert loaded.snapshots == snapshots

    def test_run_model_suppresses_generate_stdout(self, capsys) -> None:
        class FakeModel:
            def generate(self, **_kwargs):
                print("funasr version: 1.3.1.")
                return [{"text": "<|zh|> 好", "timestamp": [[0, 100]]}]

        rec = recognizer(FakeModel())

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

        fake_model = FakeModel()
        rec = recognizer(fake_model, lang="zho", snapshots=PARAFORMER)

        segments = rec._run_model("dummy.wav")

        # `return_raw_text` is what makes the timing pairable; see FunAsrSegment.
        assert fake_model.calls == [
            {"input": "dummy.wav", "output_timestamp": True, "return_raw_text": True}
        ]
        assert segments == [_S(text="好。", timestamp=[[0, 100]], raw_text="好")]

    @pytest.mark.parametrize("output", ["not-a-segment-list", [{"text": "hi"}, None]])
    def test_run_model_refuses_malformed_output(self, output) -> None:
        class FakeModel:
            def generate(self, **_kwargs):
                return output

        rec = recognizer(FakeModel())

        with pytest.raises(ModelOutputInvalidError, match="expected a dict"):
            rec._run_model("dummy.wav")


class TestFunaudioLoadEvents:
    """Time-transparency: loading the FunASR model takes seconds even from
    local disk, so ``load`` brackets it with a ``progress_v2`` pair.

    The download itself is announced (with measured progress) where it
    happens, when the worker materializes the snapshots; ``load`` reads only
    local paths. The former lazy ``_get_model`` and its "second call emits
    nothing" test are gone: a recognizer is always loaded, so there is no
    second load to suppress.
    """

    def test_load_emits_start_and_complete_events(self, monkeypatch) -> None:
        class FakeAutoModel:
            def __init__(self, **_kwargs) -> None:
                pass

        fake_funasr(monkeypatch, FakeAutoModel)
        events: list[tuple[str, object]] = []
        monkeypatch.setattr(
            "batchalign.worker._progress.write_load_progress",
            lambda stage, measure: events.append((stage, measure)),
        )

        FunAudioRecognizer.load(PARAFORMER, lang="zho", device="cpu", label="t")

        assert events == [
            ("loading_funaudio_asr", PhaseV2(edge=PhaseEdgeV2.BEGIN)),
            ("loading_funaudio_asr", PhaseV2(edge=PhaseEdgeV2.COMPLETE)),
        ]
