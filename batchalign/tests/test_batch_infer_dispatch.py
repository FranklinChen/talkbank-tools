# affects: batchalign/worker/_execute_v2.py
# affects: batchalign/inference/**
"""Tests for the shrinking legacy ``batch_infer`` dispatch boundary.

Only the text tasks keep a legacy route, and it reads the same load registry
the V2 route does. This suite keeps the legacy router honest by proving a
loaded text task's host still answers while every migrated task fails
explicitly.
"""

from __future__ import annotations

import importlib

from batchalign.tests._loaded_hosts import loads
from batchalign.worker._infer import _batch_infer, batch_infer_on
from batchalign.worker._loads import TRANSLATE_LOAD, LoadedTranslate
from batchalign.worker._types import (
    BatchInferRequest,
    BatchInferResponse,
    InferResponse,
    InferTask,
    ItemProduced,
    TaskLoad,
    TaskLoadRegistry,
    _state,
    not_loaded_message,
)

# ---------------------------------------------------------------------------
# InferTask enum coverage
# ---------------------------------------------------------------------------


class TestInferTaskEnum:
    """Verify the new InferTask variants exist and serialize correctly."""

    def test_opensmile_variant(self) -> None:
        assert InferTask.OPENSMILE.value == "opensmile"

    def test_avqi_variant(self) -> None:
        assert InferTask.AVQI.value == "avqi"

    def test_speaker_variant(self) -> None:
        assert InferTask.SPEAKER.value == "speaker"

    # Benchmark is intentionally absent here: WER scoring is Rust-only utility
    # code now, not a worker infer task.


class TestBatchItemValidation:
    """Verify Pydantic models validate batch items correctly."""

    def test_opensmile_batch_item_valid(self) -> None:
        from batchalign.inference.opensmile import OpenSmileBatchItem

        item = OpenSmileBatchItem(audio_path="/tmp/test.wav")
        assert item.audio_path == "/tmp/test.wav"

    def test_avqi_batch_item_valid(self) -> None:
        from batchalign.inference.avqi import AvqiBatchItem

        item = AvqiBatchItem(cs_file="/tmp/cs.wav", sv_file="/tmp/sv.wav")
        assert item.cs_file == "/tmp/cs.wav"
        assert item.sv_file == "/tmp/sv.wav"


# ---------------------------------------------------------------------------
# Dispatch routing (test-echo disabled, hits real dispatch)
# ---------------------------------------------------------------------------


class TestDispatchRouting:
    """Verify _batch_infer routes to correct handlers and handles errors."""

    def test_opensmile_is_not_routed_through_batch_infer_any_more(self) -> None:
        """openSMILE should now fail fast on the legacy batch-infer path."""
        req = BatchInferRequest(
            task=InferTask.OPENSMILE,
            lang="eng",
            items=[{"bad_field": "not_valid"}],
        )
        resp = _batch_infer(req)
        assert len(resp.results) == 1
        assert resp.results[0].error == (
            f"{InferTask.OPENSMILE.value} runs only through worker protocol V2"
        )

    def test_avqi_is_not_routed_through_batch_infer_any_more(self) -> None:
        """AVQI should now fail fast on the legacy batch-infer path."""
        req = BatchInferRequest(
            task=InferTask.AVQI,
            lang="eng",
            items=[{"missing": "fields"}],
        )
        resp = _batch_infer(req)
        assert len(resp.results) == 1
        assert resp.results[0].error == (
            f"{InferTask.AVQI.value} runs only through worker protocol V2"
        )

    def test_speaker_is_not_routed_through_batch_infer_any_more(self) -> None:
        req = BatchInferRequest(
            task=InferTask.SPEAKER,
            lang="eng",
            items=[{}],
        )
        resp = _batch_infer(req)
        assert len(resp.results) == 1
        assert resp.results[0].error == (
            f"{InferTask.SPEAKER.value} runs only through worker protocol V2"
        )

    def test_empty_batch_returns_empty(self) -> None:
        """Empty batch should return empty results."""
        for task in [InferTask.OPENSMILE, InferTask.AVQI]:
            req = BatchInferRequest(task=task, lang="eng", items=[])
            resp = _batch_infer(req)
            assert len(resp.results) == 0

    def test_a_loaded_text_task_answers_through_its_host(self) -> None:
        """A text task routes to the host its load recorded."""
        seen: list[object] = []

        def _fake_translate(req) -> BatchInferResponse:
            seen.append(req)
            results: list[InferResponse] = []
            for item in req.items:
                results.append(
                    InferResponse.timed(
                        lambda item=item: ItemProduced(
                            result={
                                "kind": "translated",
                                "raw_translation": str(item["text"]).upper(),
                                "engine": "test-engine",
                            }
                        )
                    )
                )
            return BatchInferResponse(results=results)

        req = BatchInferRequest(
            task=InferTask.TRANSLATE,
            lang="spa",
            items=[{"text": "hola"}],
        )
        resp = batch_infer_on(
            req, loads((TRANSLATE_LOAD, LoadedTranslate(runner=_fake_translate)))
        )

        assert len(resp.results) == 1
        assert resp.results[0].error is None
        assert resp.results[0].result == {
            "kind": "translated",
            "raw_translation": "HOLA",
            "engine": "test-engine",
        }
        # The legacy request asks for English, admitted into the one request
        # shape the V2 route also uses.
        assert len(seen) == 1
        assert seen[0].target_lang == "eng"

    def test_an_unloaded_text_task_is_refused_per_item(self) -> None:
        req = BatchInferRequest(task=InferTask.MORPHOSYNTAX, lang="eng", items=[{}, {}])

        resp = batch_infer_on(req, TaskLoadRegistry())

        # The one wording every route uses for an absent load.
        assert [result.error for result in resp.results] == [
            not_loaded_message(TaskLoad.MORPHOSYNTAX)
        ] * 2
        assert all(result.elapsed_s is None for result in resp.results)

    def test_asr_is_not_routed_through_batch_infer_any_more(self) -> None:
        """ASR should now fail fast on the legacy batch-infer path.

        The live worker uses ``execute_v2(task="asr")`` instead. Keeping this
        assertion explicit prevents the old ASR request bag from silently
        re-entering the worker bootstrap path later.
        """

        req = BatchInferRequest(
            task=InferTask.ASR,
            lang="eng",
            items=[{"audio_path": "/tmp/example.wav"}],
        )
        resp = batch_infer_on(req, TaskLoadRegistry())
        assert len(resp.results) == 1
        assert resp.results[0].error == "asr runs only through worker protocol V2"


# ---------------------------------------------------------------------------
# Capabilities advertisement
# ---------------------------------------------------------------------------


class TestCapabilitiesAdvertisement:
    """Verify _capabilities() includes new infer tasks."""

    def test_capabilities_import_gated_tasks(self) -> None:
        """Tasks with importable deps are advertised; others are excluded."""
        from batchalign.worker._types import _state

        old_test_echo = _state.test_echo
        old_ready = _state.ready
        _state.test_echo = False
        _state.ready = True
        try:
            from batchalign.worker._handlers import _capabilities

            caps = _capabilities()
            # Stanza is always installed in dev → morphosyntax, utseg, coref
            assert InferTask.MORPHOSYNTAX in caps.infer_tasks
            assert InferTask.UTSEG in caps.infer_tasks
            assert InferTask.COREF in caps.infer_tasks
            # Tasks gated on optional deps: only present if installed
            # (no assertion on OPENSMILE, AVQI, SPEAKER, ASR, may or may not be installed)
        finally:
            _state.test_echo = old_test_echo
            _state.ready = old_ready

    def test_capabilities_uses_injected_revai_key_without_config_read(
        self, monkeypatch
    ) -> None:
        """ASR capability should come from the injected bootstrap key, not config discovery."""
        import batchalign.config as legacy_config
        from batchalign.worker._handlers import _capabilities
        from batchalign.worker._types import WorkerBootstrapRuntime

        original_import_module = importlib.import_module

        def fake_import_module(name: str, package=None):
            if name == "whisper":
                raise ModuleNotFoundError(name)
            return original_import_module(name, package)

        def fail_config_read(*_args, **_kwargs):
            raise AssertionError("config_read should not run for capability detection")

        old_test_echo = _state.test_echo
        old_ready = _state.ready
        old_bootstrap = _state.bootstrap
        try:
            monkeypatch.setattr(importlib, "import_module", fake_import_module)
            monkeypatch.setattr(legacy_config, "config_read", fail_config_read)
            _state.test_echo = False
            _state.ready = False
            _state.bootstrap = WorkerBootstrapRuntime(
                task=None,
                lang="eng",
                num_speakers=1,
                revai_api_key="from-rust",
            )

            caps = _capabilities()
            assert InferTask.ASR in caps.infer_tasks
            # No ASR engine is loaded, so none is named: null, never a
            # name guessed from which credentials happen to be present.
            assert caps.engine_versions[InferTask.ASR] is None
        finally:
            _state.test_echo = old_test_echo
            _state.ready = old_ready
            _state.bootstrap = old_bootstrap
